from docutils import nodes
from docutils.parsers.rst import Directive


class YAML(Directive):

    def run(self):
        paragraph_node = nodes.paragraph(text='[YAML]')
        return [paragraph_node]


def setup(app):
    app.add_directive("yaml", YAML)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
