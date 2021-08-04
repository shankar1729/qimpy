from docutils.parsers.rst import Directive
from docutils import nodes
from typing import get_type_hints, Dict
import importlib
import inspect


class YamlDocDirective(Directive):

    has_content = True

    def run(self):

        # Split class name into module and class:
        full_class_path = self.content[0].split('.')
        module_name = '.'.join(full_class_path[:-1])
        class_name = full_class_path[-1]

        # Import module and get root class:
        rootclass = getattr(importlib.import_module(module_name), class_name)
        param_types = get_type_hints(rootclass.__init__)
        param_docs = get_parameters(rootclass.__init__.__doc__)
        signature = inspect.signature(rootclass.__init__)

        # Report only parameters which have a [YAML] in their docstring:
        nodelist = []
        for param_name, param_type in param_types.items():
            if param_name in param_docs:
                param_doc = param_docs[param_name]
                if ':yaml:' in param_doc:
                    param_doc = param_doc.replace(':yaml:', '')
                    default_value = signature.parameters[param_name].default
                    default_str = ('' if (default_value is inspect._empty)
                                   else f' = {default_value}')
                    param_text = f'{param_name} ({param_type})' \
                                 f'{default_str}:\n {param_doc}'
                    nodelist.append(nodes.paragraph(text=param_text))

        return nodelist


def get_parameters(docstr: str) -> Dict[str, str]:
    """Parse constructor docstring `docstr` into parameter descriptions."""
    result: Dict[str, str] = {}
    lines = docstr.split('\n')
    parameter_start = len(lines)
    parameter_indent = -1
    param_name = None
    param_desc = None
    for i_line, line_raw in enumerate(lines):
        line = line_raw.lstrip()
        n_indent = len(line_raw) - len(line)
        line = line.rstrip()
        # Detect start of parameters:
        if line == 'Parameters':
            parameter_start = i_line + 2
            parameter_indent = n_indent
        if (i_line < parameter_start) or (not line):
            continue
        # Parse parameters:
        if n_indent == parameter_indent:
            # Flush previous parameter, if any:
            if param_name and param_desc:
                result[param_name] = param_desc
            # Start new parameter
            param_name = line.split(':')[0].strip()
            param_desc = ''
        else:
            param_desc += ' ' + line

    # Flush last parameter:
    if param_name and param_desc:
        result[param_name] = param_desc
    return result


def yaml_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Process the :yaml: keyword in the python API documentation."""
    return [nodes.Text('[YAML]')], []


def setup(app):
    app.add_directive('yamldoc', YamlDocDirective)
    app.add_role('yaml', yaml_role)
