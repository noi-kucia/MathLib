import re
import math
from typing import List, Tuple


def pn_split_via_operator(tokens) -> Tuple:
    """ This functon returns tuple of most external operator or funtion and two or 1 argument of it
    in PN representation for instance: [- 5 * 3 sin x] -> ('-', [5], [* 3 sin x]) that represents difference of
    5 and 3sin(x)
    [ sin * ^ x 2 3 ] -> ( 'sin', [* ^ x 2 3]) that represents sinus of ( 3x^2)"""
    required_operands = 1
    if tokens[0] in Formula.functions:
        return (tokens[0], tokens[1:],)
    for i, token in enumerate(tokens[1:]):
        if type(token) in (float, int) or token == 'x':
            required_operands -= 1
        elif token in Formula.operators_precedence:
            required_operands += 1
        if required_operands == 0:
            return tokens[0], tokens[1:i + 2], tokens[i + 2:]
    raise Exception('wrong formula :(')


class Formula:
    """ The essence of this class is to translate mathematical formula from the string representation
    into a list of tokens that represents Polish notation form of it and keep it inside"""

    operators_precedence = {'-': 1, '+': 1, '*': 4, '/': 4, '%': 4, '^': 5, '(': 10, ')': 10}
    functions = {'abs': abs, 'sqrt': math.sqrt, 'rt': math.sqrt, 'exp': math.exp, 'tan': math.tan, 'tg': math.tan,
                 'sin': math.sin, 'cos': math.cos, 'log': math.log10, 'lg': math.log10,
                 'ln': lambda x: math.log(math.e, x)}
    function_precedence = 8

    def __init__(self, formula: str):
        self.tokens = self.translate_to_pn(formula)

    def __str__(self):
        for token in self.tokens:
            return f'infix: [{self.translate_to_infix()}], PN: [{" ".join(list(map(str, self.tokens)))}]'

    def translate_to_pn(self, formula: str):

        # preparing string to translate
        formula = formula.replace('\n', '')
        formula = formula.lower()
        if formula[0] == '-':
            formula = '0' + formula
        formula = re.sub(r'\(\s*-', '(0-', formula)  # adding 0 before unary '-'
        formula = re.sub(r'(\d)\s*([a-zA-Z(])', r'\1*\2', formula)  # adding '*' signs between parentheses )*(
        formula = re.sub(r'\)\(', ')*(', formula)
        formula = re.sub(r':', '/', formula)  # changing ':' to '/'
        formula = re.sub(r'\)\s*([a-zA-Z]+)', r')*\1', formula)  # adding '*' between ')' and word

        pn_formula = []
        operators_stack = []
        formula = formula[::-1]  # reversing input string !!!
        symb_i = 0  # index of the first symbol of the token we're parsing now
        while symb_i < len(formula):
            symbol_to_parse = formula[symb_i]

            if symbol_to_parse in (' ', '\t', '\a', '\r', '\v'):  # skipping whitespaces
                symb_i += 1
                continue

            if symbol_to_parse.isdigit():  # token is some number
                token = symbol_to_parse
                for next_symb in formula[symb_i + 1:]:
                    if next_symb.isdigit() or next_symb == '.':
                        token += next_symb  # adding this symbol as a part of number token
                        symb_i += 1  # going to the next symbol
                    else:
                        break
                # adding number to the output if it's correct float or int
                try:
                    token = token[::-1]
                    pn_formula.append(float(token) if '.' in token else int(token))
                except Exception:
                    raise Exception(f'number token "{token}" is wrong')

            elif symbol_to_parse in self.operators_precedence:  # token is some operator
                token = symbol_to_parse
                if token == ')':
                    operators_stack.append(token)

                elif token == '(':
                    try:
                        # pop all operators from stack until ')' is popped operator and  put them to the output
                        previous_operator = operators_stack.pop()
                        while not previous_operator == ')':
                            pn_formula.append(previous_operator)
                            previous_operator = operators_stack.pop()

                    except Exception:
                        raise Exception('one of the brackets is not closed')

                else:
                    current_operator_precedence = self.operators_precedence[token]

                    # if operator(func) on the top of stack has higher precedence, and it is not ')'
                    # then pop it into output before push current operator on stack
                    for previous_token in operators_stack[::-1]:
                        if previous_token == ')':
                            break
                        if current_operator_precedence < (
                                self.function_precedence if previous_token in self.functions else
                                self.operators_precedence[previous_token]) and not previous_token == ')':
                            operators_stack.pop()
                            pn_formula.append(previous_token)
                        else:
                            break
                    operators_stack.append(token)

            elif symbol_to_parse.isalpha():  # function, variable or constant
                token = symbol_to_parse

                for next_symb in formula[symb_i + 1:]:
                    if not next_symb.isalpha():
                        break
                    token += next_symb
                    symb_i += 1
                token = token[::-1]  # string was reversed, so we must reverse our token to get it right

                if token == 'x':
                    pn_formula.append(token)
                elif token in self.functions:
                    operators_stack.append(token)
                else:
                    raise Exception(f'unknown token "{token}"')

            symb_i += 1

        # popping all operators left
        if ')' in operators_stack:
            raise Exception('one of the brackets is not opened')
        for token in operators_stack[::-1]:
            pn_formula.append(token)

        pn_formula = list(reversed(pn_formula))  # reversing result

        return pn_formula

    def translate_to_infix(self, __remainder=None):
        if not __remainder:
            __remainder = self.tokens
        if len(__remainder) == 1:
            return str(__remainder[0])
        splitted = pn_split_via_operator(__remainder)
        if len(splitted) == 2:
            return f'{splitted[0]}({self.translate_to_infix(splitted[1])})'
        else:
            conjuction_precedence = Formula.operators_precedence[splitted[0]]
            infix = ''
            if len(splitted[1]) == 1:
                infix += str(splitted[1][0])
            elif splitted[1][0] in Formula.functions:
                infix += self.translate_to_infix(splitted[1])
            elif Formula.operators_precedence[splitted[1][0]] < conjuction_precedence:
                infix += f'({self.translate_to_infix(splitted[1])})'
            else:
                infix += f'{self.translate_to_infix(splitted[1])}'
            infix += splitted[0]
            if len(splitted[2]) == 1:
                infix += str(splitted[2][0])
            elif splitted[2][0] in Formula.functions:
                infix += self.translate_to_infix(splitted[2])
            elif Formula.operators_precedence[splitted[2][0]] < conjuction_precedence:
                infix += f'({self.translate_to_infix(splitted[2])})'
            else:
                infix += f'{self.translate_to_infix(splitted[2])}'
            return infix


print(Formula('x^2 + 3'))
print(Formula('x^3 / 3'))
print(Formula('(1+1/x)^((x^2)*3)+5'))

# TODO: сделать метод получить производную, которая будет возвращать новую формулу-производную
