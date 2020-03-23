import os

from networkx.drawing.nx_agraph import write_dot

from tree_parser.parser import DependencyParser

_path = os.path.dirname(__file__)

_save_filename = os.path.join(_path, '../data/tree_parser.model')

_text = """
In nuclear physics, the island of stability is a predicted set of isotopes of superheavy elements 
that may have considerably longer half-lives than known isotopes of these elements. 
"""

if __name__ == '__main__':
    parser = DependencyParser(_save_filename)
    g = parser.parse(_text)

    write_dot(g, 'test.dot')
