from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from docutils import nodes
from typing import get_type_hints, get_args, get_origin, \
    Dict, Sequence, List, Tuple
import importlib
import inspect


class YamlDocDirective(Directive):

    has_content = True

    def run(self) -> list:
        rootclass = get_class_from_path(self.content[0])
        constructable = get_class_from_path(self.content[1]
                                            if (len(self.content) > 1)
                                            else 'qimpy.Constructable')

        # Recursively add documentation, one entry per row:
        # --- each row has a command column and a documentation column
        rowlist = []
        self.add_class(rootclass, 0, rowlist, constructable)

        # Collect into a table:
        # --- body
        tbody = nodes.tbody()
        for row in rowlist:
            tbody += row
        # --- group
        tgroup = nodes.tgroup(cols=2)
        for colwidth in (30, 70):
            tgroup += nodes.colspec(colwidth=colwidth)
        tgroup += tbody
        # --- overall table
        table = nodes.table()
        table += tgroup
        return [table]

    def add_class(self, cls: type, level: int, rowlist: List[nodes.row],
                  constructable: type) -> None:
        """Recursively add documentation for class `cls` at depth `level`
        to `rowlist`. Here, `constructable` specifies the base class of all
        objects that are dict/yaml initializable (eg. qimpy.Constructable)."""
        # Get parameter types, documentation and default values:
        assert issubclass(cls, constructable)
        param_types = get_type_hints(cls.__init__)
        param_docs = get_parameters(cls.__init__.__doc__)
        signature = inspect.signature(cls.__init__)

        # Report only parameters which have a [YAML] in their docstring:
        for param_name, param_type in param_types.items():
            if param_name in param_docs:
                param_doc = param_docs[param_name]
                if ':yaml:' in param_doc:
                    param_doc = param_doc.replace(':yaml:', '')
                    default_value = signature.parameters[param_name].default
                    default_str = ('' if (default_value is inspect._empty)
                                   else f' {default_value}')
                    pad = '\u00A0' * (2*level)  # using non-breaking spaces
                    # Command cell:
                    cell_cmd = nodes.entry()
                    cell_cmd += nodes.paragraph(text=f'{pad}{param_name}:'
                                                     f'{default_str}')
                    # Documentation cell:
                    cell_doc = nodes.entry()
                    viewlist = ViewList()
                    for i_line, line in enumerate(param_doc.split('\n')):
                        print(i_line, line)
                        viewlist.append(line, 'memory.rst', i_line)
                    self.state.nested_parse(viewlist, 0, cell_doc)
                    # Collect row:
                    row = nodes.row()
                    row += cell_cmd
                    row += cell_doc
                    rowlist.append(row)

                    # Recur down on compound objects:
                    for cls_option in get_args(param_type):
                        if (inspect.isclass(cls_option)
                                and issubclass(cls_option, constructable)):
                            self.add_class(cls_option, level+1, rowlist,
                                           constructable)


def get_class_from_path(full_class_name: str) -> type:
    """Get class from fully-qualified name."""
    # Split class name into module and class:
    full_class_path = full_class_name.split('.')
    module_name = '.'.join(full_class_path[:-1])
    class_name = full_class_path[-1]
    # Import module and get class:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_parameters(docstr: str) -> Dict[str, str]:
    """Parse constructor docstring `docstr` into parameter descriptions."""
    result: Dict[str, str] = {}
    lines = docstr.split('\n')
    parameter_start = len(lines)
    parameter_indent = -1  # indent amount of parameter
    desc_indent = -1  # indent amount of parameter description
    param_name = None
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
    return result


def yaml_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Process the :yaml: keyword in the python API documentation."""
    return [nodes.Text('[YAML]')], []


def setup(app):
    app.add_directive('yamldoc', YamlDocDirective)
    app.add_role_to_domain('py', 'yaml', yaml_role)
