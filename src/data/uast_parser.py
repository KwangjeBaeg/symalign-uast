from UAST import UAST
import json
from tree_sitter import Tree
import re

class UASTParser():
    def __init__(self):
        self.language : str = None
        self.uast : UAST = None
        self.rules : dict = None
        self.cached_rules = dict()
        self.grammar_path = "./grammar/UAST_Grammar.json"
        self.grammar = json.load(open(self.grammar_path, "r"))
        self.rule_directory : str = "./ruleset/"
        self.rule_file_name : str = "UAST_rules_"
        self.AST : Tree = None
        # self.offset : int = None
        # self.prev_line : int = -1
        self.extracted : str = None
        self.function_info = dict()
        self.class_info = dict()
        self.user_defined_entity = {"uast_function": "self.function_info[snippet] = id",
                         "uast_class": "self.class_info[snippet] = id"}

    def set_rule_dir_path(self, path: str):
        self.rule_directory = path

    def set_grammar_path(self, path : str):
        self.grammar_path = path
        self.grammar = json.load(open(self.grammar_path, "r"))

    def set_language(self, language : str):
        self.language = language

        if (language not in self.cached_rules):
            rules_cache = json.load(open(self.rule_directory + self.rule_file_name + self.language + '.json', "r"))
            self.cached_rules[language] = rules_cache
        
        self.rules = self.cached_rules[language]


    def parse(self, AST) :
        if(self.language == None) :
            print("Language not loaded")
            return
        self.AST = AST
        self.uast = UAST()
        self.uast.root.metadata["language"] = self.language
        self._dfs(AST_node = self.AST.root_node, parent = self.uast.root)
        return self.uast

    def _add_user_defined(self, node):
        id = node.id
        type = node.node_type

        snippet = node.code_snippet.replace(type, '').strip()
        if (type in self.user_defined_entity):
            exec(self.user_defined_entity[type])
            node.metadata["user_defined"] = True

        del id
        del type
        del snippet
        return

    def _dfs(self, AST_node, parent, prev_line = -1, prev_offset = 0, prev_node = None) :
        if (AST_node.type in self.rules) :
            ast_snippet = AST_node.text.decode("utf8")
            node_type = self.rules[AST_node.type]["uast_node_type"]
            exec_string = self.rules[AST_node.type]["extractor"]
            uast_snippet = self._extract(ast_snippet = ast_snippet, node_type = node_type, exec_string = exec_string)

            # start_line = AST_node.start_point[0]
            # start_byte = AST_node.start_point[1]
            # end_line = AST_node.end_point[0]
            # end_byte = AST_node.end_point[1]

            # # keeping track of the prev line and the previous offset associated to that line, we add it into the current one
            # if(start_line == prev_line):
            #     start_byte += prev_offset
            # else:
            #     # done to null the offset, as a different line would not be affected by the previous changes == (ASSUMPTION)
            #     prev_offset = 0
            
            # prev_line = end_line
            # prev_offset = prev_offset + len(uast_snippet) - len(ast_snippet)

            # # end byte now has the global offset, for that line
            # end_byte += prev_offset

            node = self.uast.create_node(
                node_type = node_type,
                code_snippet = uast_snippet,
                metadata = {
                    "original_code" : ast_snippet
                },
                # start_point = (start_line, start_byte),
                # end_point= (end_line, end_byte)
            )
            self._add_user_defined(node)
            self.uast.add_edge(node1 = parent, node2 = node, directed_relation = "parent_node")
            parent = node

        for child in AST_node.children:
            self._dfs(AST_node= child, parent = parent, prev_line = prev_line, prev_offset = prev_offset)

    def _extract(self, ast_snippet, node_type, exec_string):
        code_snippet = ast_snippet
        try:
            exec(exec_string)
        except Exception as e:
            print(e)
        try:
            return self.grammar[node_type]["keyword"] + " " + self.extracted        
        except Exception as e:
            print(e)