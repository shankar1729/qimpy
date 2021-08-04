from docutils.parsers.rst import Directive
from docutils import nodes


class yamldoc(nodes.General, nodes.Element):
    pass


class YamlDocDirective(Directive):
    def run(self):
        return [yamldoc('')]


def process_yamldoc(app, doctree, fromdocname):
    # Get list of methods:
    methodnodes = dict()
    py = app.builder.env.get_domain('py')
    for name, dispname, objtype, docname, anchor, priority in py.get_objects():
        if objtype == 'method':
            newnode = nodes.reference('', '')
            newnode['refdocname'] = docname
            newnode['refuri'] = app.builder.get_relative_uri(
                fromdocname, docname) + '#' + anchor
            newnode.append(nodes.Text(name + ' ' + dispname))
            methodnodes[name.lower()] = newnode

    # A paragraph for each:
    paragraphs = []
    for key, value in sorted(methodnodes.items()):
        para = nodes.paragraph()
        para += value
        paragraphs.append(para)

    for node in doctree.traverse(yamldoc):
        node.replace_self(paragraphs)


def setup(app):
    app.add_node(yamldoc)
    app.add_directive('yamldoc', YamlDocDirective)
    app.connect('doctree-resolved', process_yamldoc)
