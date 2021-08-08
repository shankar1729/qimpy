from __future__ import annotations
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from docutils.statemachine import ViewList
from docutils import nodes
from typing import get_type_hints, get_args, get_origin, \
    Dict, List, Tuple, Optional, NamedTuple
from collections.abc import Sequence
from functools import lru_cache
import importlib
import inspect
import os


class Parameter(NamedTuple):
    """Parameter within `ClassInputDoc`."""
    name: str  #: Parameter name
    default: str  #: String representing default value, if any
    summary: str  #: One-line summary
    doc: str  #: Full doc-string.
    classdoc: Optional[ClassInputDoc] = None  #: Documentation of this class
    typename: str = ''  # Name of type (used only if no `classdoc`)


class ClassInputDoc:
    """Input documentation extracted from a `constructable` subclass."""

    def __init__(self, cls: type, constructable: type, app: Sphinx,
                 classdocs: Dict[str, ClassInputDoc], outdir: str) -> None:
        """Write input documentation for `cls` in ReST format within `outdir`.
        Recurs down to parameters marked with :yaml: in `cls.__init__` that
        are also sub-classes of `constructable`. Also adds `self` to classdocs
        to avoid writing `cls`'s documentation multiple times for overlapping
        input documentation trees."""
        self.cls = cls
        self.params: List[Parameter] = []
        self.path = get_path_from_class(cls)
        classdocs[self.path] = self

        assert issubclass(cls, constructable)
        param_types = get_type_hints(cls.__init__)
        header, param_docs = get_parameters(cls.__init__.__doc__)
        signature = inspect.signature(cls.__init__)

        # Process only parameters which have a [YAML] in their docstring:
        for param_name, param_type in param_types.items():
            if param_name in param_docs:
                param_doc = param_docs[param_name]
                if ':yaml:' in param_doc:
                    default_value = signature.parameters[param_name].default
                    default_str = ('' if (default_value is inspect._empty)
                                   else f' {default_value}')
                    param_doc, param_summary = yaml_remove_split(param_doc)
                    typenames = []
                    # Recur down on compound objects:
                    param_class: Optional[ClassInputDoc] = None
                    for cls_option in get_args(param_type):
                        if (inspect.isclass(cls_option)
                                and issubclass(cls_option, constructable)):
                            # Check if already documented cls_option:
                            cls_option_path = get_path_from_class(cls_option)
                            param_class = classdocs.get(cls_option_path, None)
                            # If not, document it now:
                            if param_class is None:
                                param_class = ClassInputDoc(cls_option,
                                                            constructable, app,
                                                            classdocs, outdir)
                        else:
                            typenames.append(yamltype(cls_option))
                    if not typenames:
                        typenames = [yamltype(param_type)]
                    typename = ' or '.join(typenames).rstrip(',')
                    param = Parameter(name=param_name.replace('_', '-'),
                                      default=yamlify(default_str),
                                      summary=param_summary,
                                      doc=yamlify(param_doc),
                                      classdoc=param_class,
                                      typename=typename)
                    self.params.append(param)

        # Helper to write a title:
        def write_title(file, title: str, underline: str) -> None:
            file.write(f'{title}\n{underline * len(title)}\n\n')

        # Write ReST file:
        fname = os.path.join(outdir, self.path + '.rst')
        with open(fname, 'w') as fp:
            # Title:
            write_title(fp, f'{cls.__qualname__} input documentation', '=')
            # Constructor header:
            fp.write(header)
            fp.write(f'\n\nUsed to initialize class :class:`{self.path}`.\n\n')
            # Template:
            write_title(fp, 'YAML template:', '-')
            fp.write('.. parsed-literal::\n\n')
            for line in self.get_yaml_template():
                fp.write(f'   {line}\n')
            fp.write('\n')
            # Component classes:
            component_classdocs = [param.classdoc for param in self.params
                                   if (param.classdoc is not None)]
            if component_classdocs:
                write_title(fp, 'Component classes:', '-')
                fp.write('.. toctree::\n    :maxdepth: 1\n\n')
                for classdoc in component_classdocs:
                    fp.write(f'    {classdoc.cls.__qualname__}'
                             f' <{classdoc.path}>\n')
                fp.write('\n')
            # Parameter detailed docs:
            write_title(fp, 'Parameters:', '-')
            for param in self.params:
                write_title(fp, param.name, '+')
                if param.default:
                    fp.write(f'*Default:* {param.default}\n\n')
                if param.classdoc is None:
                    fp.write(f'*Type:* {param.typename}\n\n')
                else:
                    fp.write(f'*Type:* :doc:`{param.classdoc.cls.__qualname__}'
                             f' <{param.classdoc.path}>`\n\n')
                fp.write(param.doc)
                fp.write('\n\n')

    @lru_cache
    def get_yaml_template(self, linkprefix='') -> List[str]:
        """Return lines of a yaml template based on parameters.
        Recursively includes templates of component classes within."""
        result = []
        for param in self.params:
            name = f':yamlparam:`{linkprefix}{self.path}:{param.name}`'
            value = ((param.default if param.default
                      else f' [{param.typename}]')
                     if (param.classdoc is None)
                     else '')  # don't put value if class doc follows
            comment = f'  :yamlcomment:`# {param.summary}`'
            result.append(f'{name}:{value}{comment}')
            # Recur down to components:
            if param.classdoc is not None:
                pad = '  '
                for line in param.classdoc.get_yaml_template():
                    result.append(pad + line)  # indent component template
        return result


class YamlDocDirective(SphinxDirective):
    """Directive that places YAML template in the source rst file.
    Note that the content has already been prepared during
    `create_yamldoc_rst_files`, and this class only needs to
    repeat that content at the source location."""
    has_content = True

    def run(self) -> list:
        class_path = self.content[0]
        classdoc: ClassInputDoc = self.env.yamldoc_classdocs[class_path]

        # Read pre-created ReST for this class into a view list:
        viewlist = ViewList()
        viewlist.append('.. parsed-literal::', 'memory.rst', 0)
        viewlist.append('', 'memory.rst', 1)
        for i_line, line in enumerate(classdoc.get_yaml_template()):
            viewlist.append('   ' + line, 'memory.rst', i_line + 2)

        # Add TOC to include root class:
        toclines = f'\nComponent classes:\n\n' \
                   f'.. toctree::\n' \
                   f'    :maxdepth: 1\n\n' \
                   f'    {classdoc.cls.__qualname__} <yamldoc/{class_path}>\n'
        for line in toclines.split('\n'):
            viewlist.append(line, 'memory.rst', len(viewlist))

        # Parse:
        node = nodes.paragraph()
        self.state.nested_parse(viewlist, 0, node)
        return [node]


def create_yamldoc_rst_files(app: Sphinx) -> None:
    """Create rst files for each class connected to any yamldoc root class."""
    # Find the doc source files (excluding autogenerated ones):
    env = app.builder.env
    docfiles = [env.doc2path(docname)
                for docname in env.found_docs
                if ((not docname.startswith(('api/', 'yamldoc/')))
                    and os.path.isfile(env.doc2path(docname)))]

    # Find all instances of the yamldoc directive:
    yamldoc_classnames = set()
    directive_key = '.. yamldoc::'
    for docfile in docfiles:
        for line in open(docfile):
            i_start = line.find(directive_key)
            if i_start >= 0:
                tokens = line[(i_start + len(directive_key)):].split()
                yamldoc_classnames.add(tokens[0])
                os.utime(docfile)  # make sure file processed in this build
    print('[yamldoc] generating input documentation for root class(es):',
          ', '.join(yamldoc_classnames))

    # Prepare directory for yamldoc class documentations:
    yamldoc_dir = os.path.join(env.srcdir, 'yamldoc')
    if not os.path.isdir(yamldoc_dir):
        os.mkdir(yamldoc_dir)

    # Document all constructable classes recursively:
    constructable = get_class_from_path('qimpy.Constructable')
    classdocs: Dict[str, ClassInputDoc] = {}
    for classname in yamldoc_classnames:
        cls = get_class_from_path(classname)
        ClassInputDoc(cls, constructable, app, classdocs, yamldoc_dir)
    env.yamldoc_classdocs = classdocs  # save for use in the directive


def get_class_from_path(full_class_name: str) -> type:
    """Get class from fully-qualified name."""
    # Split class name into module and class:
    full_class_path = full_class_name.split('.')
    module_name = '.'.join(full_class_path[:-1])
    class_name = full_class_path[-1]
    # Import module and get class:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_path_from_class(cls: type) -> str:
    """Get full path of class, getting rid of internal module names."""
    module = cls.__module__
    module_elems = ([] if module is None else (
        [elem for elem in module.split('.')
         if not elem.startswith('_')]))  # drop internal module names
    module_elems.append(cls.__qualname__)
    return '.'.join(module_elems)


def get_parameters(docstr: str) -> Tuple[str, Dict[str, str]]:
    """Parse constructor docstring `docstr` into parameter descriptions.
    Returns header of constructor documentation, and for each parameter."""
    result: Dict[str, str] = {}
    lines = docstr.split('\n')
    parameter_start = len(lines)
    parameter_indent = -1  # indent amount of parameter
    desc_indent = -1  # indent amount of parameter description
    param_name = None
    header = []
    param_desc = []
    for i_line, line_raw in enumerate(lines):
        line = line_raw.lstrip()
        n_indent = len(line_raw) - len(line)
        line = line.rstrip()
        # Detect start of parameters:
        if line == 'Parameters':
            parameter_start = i_line + 2
            parameter_indent = n_indent
        if i_line < parameter_start:
            header.append(line)
            continue
        # Parse parameters:
        if n_indent == parameter_indent:
            # Flush previous parameter, if any:
            if param_name and param_desc:
                result[param_name] = '\n'.join(param_desc).strip('\n')
            # Start new parameter
            param_name = line.split(':')[0].strip()
            param_desc = []
            desc_indent = -1
        elif line:
            if desc_indent == -1:
                desc_indent = n_indent  # based on first line in this desc
            rel_indent = n_indent - desc_indent
            param_desc.append(' '*rel_indent + line)
        else:
            param_desc.append('')  # Blank lines important in ReST formatting

    # Flush last parameter:
    if param_name and param_desc:
        result[param_name] = '\n'.join(param_desc).strip('\n')
    return '\n'.join(header[:-2]), result


PY_TO_YAML = {
    'None': ':yamlkey:`null`',
    'True': ':yamlkey:`yes`',
    'False': ':yamlkey:`no`'
}


YAML_TYPE = {
    str: ':yamltype:`string`',
    float: ':yamltype:`float`',
    int: ':yamltype:`int`',
    bool: ':yamltype:`bool`',
    list: ':yamltype:`list`',
    dict: ':yamltype:`dict`',
    tuple: ':yamltype:`tuple`',
    type(None): ':yamltype:`null`'
}


SEQUENCE_TYPES = {list, tuple, Sequence}


def yamlify(doc: str) -> str:
    """Replace python keywords with yaml versions in `doc`"""
    for py_word, yaml_word in PY_TO_YAML.items():
        doc = doc.replace(py_word, yaml_word)
    return doc


def yamltype(cls: type) -> str:
    """Return YAML name for type `cls`."""
    result = YAML_TYPE.get(cls, None)
    if result is not None:
        return result
    # Check for sequence types:
    origin = get_origin(cls)
    if origin in SEQUENCE_TYPES:
        result = ':yamltype:`list`'
        args = get_args(cls)
        delim = ' and ' if (origin is tuple) else ' or '
        if len(args):
            result += ' of ' + delim.join(yamltype(arg) for arg in args) + ','
        return result
    # Fallback to python name:
    return str(cls)


def yaml_remove_split(docstr: str) -> Tuple[str, str]:
    """Extract parameter summary within :yaml: tags in docstring,
    and clean up the :yaml: tag for the full docstring.
    Return cleaned up docstring, and summary version."""
    key = ':yaml:'
    summary = ''
    i_start = docstr.find(key)
    while i_start >= 0:
        i_key_stop = i_start + len(key) + 1  # end of :yaml:
        i_stop = docstr.find('`', i_key_stop) + 1  # end of content after it
        fullkey = docstr[i_start:i_stop]  # includes `content` after key
        summary = docstr[i_key_stop:(i_stop-1)]  # just the content
        docstr = docstr.replace(fullkey, summary)
        # Search for any other keys:
        i_start = docstr.find(key)
    return docstr, summary


def yaml_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link :yaml: in the python API docs to the input file docs."""
    env = inliner.document.settings.env
    app: Sphinx = env.app
    src_doc = env.docname
    dest_doc = src_doc.replace('api/', 'yamldoc/')
    uri = app.builder.get_relative_uri(src_doc, dest_doc)
    return [nodes.reference(rawtext, '[Input file]', refuri=uri, **options),
            nodes.Text(' ' + text)], []


def yaml_param_role(name, rawtext, text, lineno, inliner,
                    options={}, content=[]):
    """Link :yamlparam: in the input file doc to detailed version."""
    env = inliner.document.settings.env
    app: Sphinx = env.app
    dest_doc, param_name = text.split(':')
    uri = (app.builder.get_relative_uri(env.docname, 'yamldoc/' + dest_doc)
           + '#' + param_name)
    return [nodes.reference(rawtext, param_name,
                            refuri=uri, classes=['yamlparam'], **options)], []


def yaml_highlight(rolename: str):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        return [nodes.inline(text=text, classes=['yaml'+rolename])], []
    return role


def setup(app):
    app.add_directive('yamldoc', YamlDocDirective)
    app.add_role_to_domain('py', 'yaml', yaml_role)
    app.add_role('yamlparam', yaml_param_role)
    app.add_role('yamlkey', yaml_highlight('key'))
    app.add_role('yamltype', yaml_highlight('type'))
    app.add_role('yamlcomment', yaml_highlight('comment'))
    app.connect('builder-inited', create_yamldoc_rst_files)
