import builtins
import re
import math
from typing import List, Tuple


def gcd(a, b):
    """
    returns the greatest common divisor of a and b using Euclidean algorithm.
    if at least 1 of arguments isn't integer, returns 1
    """
    if type(a) != int or type(b) != int:
        return 1
    if a == 0:
        return b
    elif b == 0:
        return a

    while b != 0:
        temp = b
        b = a % b
        a = temp

    return a


def get_precedence(token):
    """returns the precedence of the operator/function from Formula class static values"""
    if token in Formula.functions:
        return Formula.function_precedence
    elif token in Formula.operators_precedence:
        return Formula.operators_precedence[token]
    else:
        raise Exception(f'token {token} is not known for Formula class!')


def get_operator(tokens: List):
    """
    returns operator/functor of expression if it's composite, otherwise returns empty string
    """
    if len(tokens) == 1:
        return ''
    return tokens[0]


def pn_to_latex(tokens: List):
    """
    :param tokens: polish notation sequence
    :return: latex (MathJax) form of expression
    """
    if len(tokens) == 1:
        return str(tokens[0])
    operator, arg1, arg2 = pn_split_via_operator(tokens)

    if operator in Formula.functions:
        if operator == 'exp':
            return r'e^{' + pn_to_latex(arg1) + '}'
        if operator == 'sqrt':
            return r'\sqrt{' + pn_to_latex(arg1) + '}'
        return f'\\{operator}\\left({pn_to_latex(arg1)}\\right)'

    elif operator in Formula.operators_precedence:

        if operator == '/':
            return '\\frac{' + pn_to_latex(arg1) + '}{' + pn_to_latex(arg2) + '}'
        if operator == '^':
            if len(arg1) == 1 or get_precedence(arg1[0]) > get_precedence('^'):
                return pn_to_latex(arg1) + '^{' + pn_to_latex(arg2) + '}'
            else:
                return f'({pn_to_latex(arg1)})' + '^{' + pn_to_latex(arg2) + '}'
        if operator == '+':
            if is_scalar_pn(arg1) and arg1[0] < 0:
                arg1, arg2 = arg2, [-1*arg1[0]]
                operator = '-'
            elif is_scalar_pn(arg2) and arg2[0] < 0:
                arg2 = [-1 * arg2[0]]
                operator = '-'
            if arg2[0] == '*':
                _, product_arg1, product_arg2 = pn_split_via_operator(arg2)
                if is_scalar_pn(product_arg1) and product_arg1[0] < 0:
                    arg2 = ['*', -1*product_arg1[0]] + product_arg2
                    operator = '-'
            elif arg2[0] == '/':
                _, quotient_arg1, quotient_arg2 = pn_split_via_operator(arg2)
                if is_scalar_pn(quotient_arg1) and quotient_arg1[0] < 0:
                    arg2 = ['/'] + [-1*quotient_arg1[0]] + quotient_arg2
                    return f'{pn_to_latex(arg1)}-{pn_to_latex(arg2)}'

        if operator == '*' and is_scalar_pn(arg1) and arg1[0] == [-1]:
            arg2_precedence = get_precedence(arg2[0]) if not len(arg2) == 1 else 20
            if arg2_precedence > get_precedence('*'):
                return f'-{pn_to_latex(arg2)}'
            else:
                return f'-({pn_to_latex(arg2)})'

        operator_precedence = get_precedence(operator)
        latex = ''
        if len(arg1) == 1:
            oper1_prec = 20
        else:
            oper1_prec = get_precedence(pn_split_via_operator(arg1)[0])\
                if pn_split_via_operator(arg1)[0] not in Formula.functions\
                else Formula.function_precedence
        if len(arg2) == 1:
            oper2_prec = 20
        else:
            oper2_prec = get_precedence(pn_split_via_operator(arg2)[0])\
                if pn_split_via_operator(arg2)[0] not in Formula.functions\
                else Formula.function_precedence
        if oper1_prec < operator_precedence:
            latex += f'\\left({pn_to_latex(arg1)}\\right)'
        else:
            latex += pn_to_latex(arg1)
        latex += operator
        if oper2_prec < operator_precedence:
            latex += f'\\left({pn_to_latex(arg2)}\\right)'
        else:
            latex += pn_to_latex(arg2)
        return latex

    else:
        raise Exception('Unable to convert into LaTex')


class UnableToDifferentiateException(Exception):
    def __init__(self, message='cannot find derivative!'):
        super().__init__(message)


def pn_split_via_operator(tokens) -> Tuple:
    """
    This functon returns tuple of most external operator or function and two or 1 argument of it
    in PN representation. If it's function, None will be returned as second argument.
    For instance:
        [- 5 * 3 sin x] -> ('-', [5], [* 3 sin x]) that represents difference of 5 and 3sin(x)
        [ sin * ^ x 2 3 ] -> ( 'sin', [* ^ x 2 3], None) that represents sinus of ( 3x^2)
    """
    required_operands = 1
    if tokens[0] in Formula.functions:
        return (tokens[0], tokens[1:], None)
    for i, token in enumerate(tokens[1:]):
        if type(token) in (float, int) or token == 'x':
            required_operands -= 1
        elif token in Formula.operators_precedence:
            required_operands += 1
        if required_operands == 0:
            return tokens[0], tokens[1:i + 2], tokens[i + 2:]
    raise Exception('wrong formula :(')


class Formula:
    """
    The essence of this class is to translate mathematical formula from the string representation
    into a list of tokens that represents Polish notation form of it and keep it inside
    """

    operators_precedence = {'-': 1, '+': 1, '*': 4, '/': 4, '%': 4, '^': 6, '(': 10, ')': 10}
    functions = {'abs': abs, 'sqrt': math.sqrt, 'rt': math.sqrt, 'exp': math.exp, 'tan': math.tan, 'tg': math.tan,
                 'sin': math.sin, 'cos': math.cos, 'log': math.log10, 'lg': math.log10, 'arccos': math.acos,
                 'ln': lambda x: math.log(math.e, x), 'ctg': lambda x: 1./math.tan(x), 'arcsin': math.asin,
                 'asin': math.asin, 'acos': math.acos, 'arctg': math.atan, 'arcctg': lambda x: -1*math.atan(x)}
    function_precedence = 8

    def __init__(self, formula):
        match type(formula):
            case builtins.list:
                self.tokens = formula
            case builtins.str:
                self.tokens = self.translate_to_pn(formula)
            case _:
                raise Exception(f'unknown input type: {type(formula)}')

    def __str__(self):
        for token in self.tokens:
            return f'[{self.translate_to_infix()}]'

    def get_pn_string(self) -> str:
        """Returns formula in polish notation"""
        return " ".join(list(map(str, self.tokens)))

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
                    current_operator_precedence = get_precedence(token)

                    # if operator(func) on the top of stack has higher precedence, and it is not ')'
                    # then pop it into output before push current operator on stack
                    for previous_token in operators_stack[::-1]:
                        if previous_token == ')':
                            break
                        if current_operator_precedence < get_precedence(previous_token) and not previous_token == ')':
                            operators_stack.pop()
                            pn_formula.append(previous_token)
                        else:
                            break
                    operators_stack.append(token)

            elif symbol_to_parse.isalpha():  # function, variable or scalar
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
                    while operators_stack and self.operators_precedence[operators_stack[-1]] > self.function_precedence \
                            and operators_stack[-1] != ')':
                        pn_formula.append(operators_stack.pop())
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

        return pn_prettify(pn_formula)

    def translate_to_infix(self, __remainder=None):
        if not __remainder:
            __remainder = self.tokens
        if len(__remainder) == 1:
            return str(__remainder[0])
        operator, arg1, arg2 = pn_split_via_operator(__remainder)
        if operator in Formula.functions:
            return f'{operator}({self.translate_to_infix(arg1)})'
        else:
            conjunction_precedence = Formula.operators_precedence[operator]
            infix = ''
            if operator == '/' and len(arg2) > 1:
                return f'{self.translate_to_infix(arg1)}/({self.translate_to_infix(arg2)})'
            if operator == '+':
                if is_scalar_pn(arg1):
                    arg1, arg2 = arg2, arg1

                if is_scalar_pn(arg2):
                    if arg2[0] < 0:
                        return f'{arg1}-{-1*arg2[0]}'
                elif arg2[0] == '*':
                    _, product_arg1, product_arg2 = pn_split_via_operator(arg2)
                    if is_scalar_pn(product_arg1) and product_arg1[0] < 0:
                        return f'{self.translate_to_infix(arg1)}-' \
                               f'{-1*product_arg1[0]}*{self.translate_to_infix(product_arg2)}'

            if len(arg1) == 1:
                infix += str(arg1[0])
            elif arg1[0] in Formula.functions:
                infix += self.translate_to_infix(arg1)
            elif Formula.operators_precedence[arg1[0]] < conjunction_precedence:
                infix += f'({self.translate_to_infix(arg1)})'
            else:
                infix += f'{self.translate_to_infix(arg1)}'
            infix += operator
            if len(arg2) == 1:
                infix += str(arg2[0])
            elif arg2[0] in Formula.functions:
                infix += self.translate_to_infix(arg2)
            elif Formula.operators_precedence[arg2[0]] < conjunction_precedence:
                infix += f'({self.translate_to_infix(arg2)})'
            else:
                infix += f'{self.translate_to_infix(arg2)}'
            return infix


def is_scalar_pn(tokens: List):
    return True if type(tokens[0]) in (int, float) else False


def is_pn_single_var(tokens: List):
    return True if len(tokens) == 1 and tokens[0] == 'x' else False


def is_constant_pn(tokens: List):
    """ returns True only if expression is some constant value (independent of x)"""
    if not 'x' in tokens:
        return True
    return False


def pn_prettify(tokens: List):
    """
    :param tokens:
    :return: mathematically equal formula in optimized form
    It deletes all '+0' or '*1' unnecessary operations, sums all 'scalar+scalar' expressions to a single token
    and so on. If it cannot optimize expression, it'll return it back in the same form.
    """
    if len(tokens) == 1:
        return tokens

    operator, arg1, arg2 = pn_split_via_operator(tokens)

    if operator in Formula.operators_precedence:
        match operator:
            case '+':
                arg1, arg2 = pn_prettify(arg1), pn_prettify(arg2)

                if len(arg1) == 1:
                    if arg1[0] == 0:
                        return arg2
                if len(arg2) == 1:
                    if arg2[0] == 0:
                        return arg1

                if is_constant_pn(arg1) and is_constant_pn(arg2):
                    if is_scalar_pn(arg1) and is_scalar_pn(arg2):  # if both summand are scalars, we can calculate it
                        return [arg1[0] + arg2[0]]

                    if get_operator(arg2) == '/':
                        arg1, arg2 = arg2, arg1

                    if get_operator(arg1) == '/':
                        _, numerator, denominator = pn_split_via_operator(arg1)
                        if type(numerator[0]) == int and type(denominator[0]) == int:
                            if get_operator(arg2) == '/':
                                _, nominator2, denominator2 = pn_split_via_operator(arg2)
                                if type(nominator2[0]) == int and type(denominator2[0]) == int:
                                    return pn_prettify(['/', numerator[0]*denominator2[0]+nominator2[0]*denominator[0],
                                                        denominator[0]*denominator2[0]])

                            else:
                                return pn_prettify(['/', numerator[0]+arg2[0]*denominator[0], denominator[0]])

                if arg1 == arg2:
                    return pn_prettify(['*', 2] + arg1)

                if get_operator(arg2) == '*' and arg1[0] != '*':
                    # since now, if there's a product in sum, it will be on the first place
                    arg1, arg2 = arg2, arg1

                if get_operator(arg1) == '*':
                    _, product_arg1, product_arg2 = pn_split_via_operator(arg1)

                    if get_operator(arg2) == '*':  # sum of two products #TODO: solve cases like a*b + c*b*d = b(a+c*d) and e.g.
                        _, product2_arg1, product2_arg2 = pn_split_via_operator(arg2)
                        if is_scalar_pn(product_arg1) and is_scalar_pn(product2_arg1):
                            if product_arg2 == product2_arg2:
                                return pn_prettify(
                                    ['*'] + [product_arg1[0]+product2_arg1[0]] + ['+'] + product_arg2 + product2_arg2)

                    elif is_scalar_pn(product_arg1):  # sum has structure constant*expr1 + expr2
                        if product_arg2 == arg2:  # constant*x + x = (constant + 1)x
                            return pn_prettify(['*', product_arg1[0] + 1] + arg2)
                        elif product_arg1[0] < 0:  # -constant*expr1 + expr2 = expr2 - constant*expr1
                            return ['-'] + arg2 + pn_prettify(['*', -1*product_arg1[0]] + product_arg2)

                return ['+'] + arg1 + arg2

            case '-':
                # subtraction is equal to addition
                return pn_prettify(['+'] + arg1 + ['*', -1] + arg2)

            case '*':
                arg1, arg2 = pn_prettify(arg1), pn_prettify(arg2)

                if is_scalar_pn(arg2):  # if one of multipliers is scalar, it must be on the first place
                    arg1, arg2 = arg2, arg1

                if not is_scalar_pn(arg1):  # if any of multipliers ain't scalars

                    if arg1 == arg2:  # x*x = x^2
                        return ['^'] + arg1 + [2]

                    if get_operator(arg2) == '/':
                        _, numerator, denominator = pn_split_via_operator(arg2)
                        if numerator == [1]:
                            return pn_prettify()
                        if numerator == [-1]:
                            return pn_prettify(['*', -1, '/'] + arg1 + denominator)
                    if get_operator(arg1) == '/':
                        _, numerator, denominator = pn_split_via_operator(arg1)
                        if numerator == [1]:
                            return pn_prettify(['/'] + arg2 + denominator)
                        if numerator == [-1]:
                            return pn_prettify(['*', -1, '/'] + arg2 + denominator)

                    if get_operator(arg2) == '*':  # if one of multipliers is product, it must be on the first place
                        arg1, arg2 = arg2, arg1

                    if get_operator(arg1) == '*':
                        _, arg1_scalar, arg1_expr = pn_split_via_operator(arg1)
                        if get_operator(arg2) == '*':
                            _, arg2_scalar, arg2_expr = pn_split_via_operator(arg2)
                            if is_scalar_pn(arg1_scalar):
                                if is_scalar_pn(arg2_scalar):
                                    return ['*', arg1_scalar[0] * arg2_scalar[0], '*'] + arg1_expr + arg2_expr
                                return ['*'] + arg1_scalar + ['*'] + arg1_expr + arg2
                            elif is_scalar_pn(arg2_scalar):
                                return ['*'] + arg2_scalar + ['*'] + arg1 + arg2_expr
                        elif is_scalar_pn(arg1_scalar):
                            return ['*'] + arg1_scalar + ['*'] + arg1_expr + arg2
                        else:
                            return ['*'] + arg1 + arg2

                    if get_operator(arg2) == '^':
                        arg1, arg2 = arg2, arg1

                    if get_operator(arg1) == '^':
                        _, base, exponent = pn_split_via_operator(arg1)
                        if get_operator(arg2) == '^':
                            _, base2, exponent2 = pn_split_via_operator(arg2)
                            if base == base2:
                                return ['^'] + base + pn_prettify(['*'] + exponent + exponent2)
                        if base == arg2:
                            return ['^'] + base + pn_prettify(['+'] + exponent + [1])

                    return ['*'] + arg1 + arg2

                if is_scalar_pn(arg2):
                    # product of two scalars is scalar
                    return [arg1[0] * arg2[0]]

                # since this point, we have product of scalar and non-scalar expression ( scalar*expression )
                scalar, expr = arg1[0], arg2

                if scalar == 1:  # 1*expr = expr
                    return expr
                if scalar == 0:  # 0*expr = 0
                    return [0]
                if expr == ['x']:  # scalar*x
                    return ['*', scalar, 'x']

                # now splitting expression
                expression_operator, expression_arg1, expression_arg2 = pn_split_via_operator(expr)
                match expression_operator:
                    case '*':
                        if is_scalar_pn(expression_arg1):
                            return ['*', arg1[0] * expression_arg1[0]] + expression_arg2
                    case '/':
                        numerator, denominator = expression_arg1, expression_arg2

                        if is_scalar_pn(numerator):
                            return ['/', scalar] + denominator  # scalar*(scalar/expr) = scalar * expr
                        if numerator == ['x']:
                            return ['*'] + arg1 + arg2

                        nomin_operator, nomin_arg1, nomin_arg2 = pn_split_via_operator(numerator)
                        if nomin_operator == '*' and is_scalar_pn(nomin_arg1):
                            #  scalar * [(scalar*expr)/ expr]= (scalar*expr)/expr
                            return ['/', '*', scalar * nomin_arg1[0]] + nomin_arg2 + denominator

                return ['*'] + arg1 + arg2

            case '/':
                numerator, denominator = pn_prettify(arg1), pn_prettify(arg2)

                if numerator == [0]:
                    return [0]
                if denominator == [1]:
                    return numerator

                if is_scalar_pn(numerator):  # numerator is scalar
                    if len(denominator) > 2:  # denominator is composite expression
                        denominator_operator, denominator_arg1, denominator_arg2 = pn_split_via_operator(denominator)

                        if denominator_operator == '*':
                            if is_scalar_pn(denominator_arg1):
                                divisor = gcd(numerator[0], denominator_arg1[0])
                                if divisor > 1:
                                    if divisor == denominator_arg1[0]:
                                        return ['/', numerator[0] // divisor] + denominator_arg2
                                    return ['/', numerator[0] // divisor,
                                            '*', denominator_arg1[0] // divisor] + denominator_arg2
                    elif is_scalar_pn(denominator):
                        divisor = gcd(numerator[0], denominator[0])
                        if divisor > 1:
                            return ['/', numerator[0] // divisor, denominator[0] // divisor]

                if len(numerator) > 2:  # numerator is some composite expression
                    nominator_operator, nominator_arg1, nominator_arg2 = pn_split_via_operator(numerator)

                    # (expression/scalar)/scalar = expression/scalar
                    if is_scalar_pn(denominator):
                        if nominator_operator == '/' and is_scalar_pn(nominator_arg2):
                            if type(denominator[0]) == int and type(nominator_arg2[0]) == int:  # to avoid float error
                                return ['/'] + nominator_arg1 + [denominator[0] * nominator_arg2[0]]

                return ['/'] + numerator + denominator

            case '^':
                arg1, arg2 = pn_prettify(arg1), pn_prettify(arg2)

                if is_scalar_pn(arg1) and is_scalar_pn(arg2):
                    return pow(arg1[0], arg2[0])

                if len(arg1) == 1:
                    if arg1[0] == 1:
                        return [1]
                else:
                    if get_operator(arg1) == '^':
                        _, base_arg1, base_arg2 = pn_split_via_operator(arg1)
                        return ['^'] + base_arg1 + ['*'] + base_arg2 + arg2

                if len(arg2) == 1:
                    if arg2[0] == 1:
                        return arg1
                    if arg2[0] == 0:
                        return [1]

                return ['^'] + arg1 + arg2

    elif operator in Formula.functions:
        arg1 = pn_prettify(arg1)

        if operator == 'exp':
            if len(arg1) > 2:
                exp_operator, exp_arg1, exp_arg2 = pn_split_via_operator(arg1)
                if exp_operator == '*':
                    if exp_arg1[0] == 'ln':
                        return ['^'] + exp_arg1[1:] + exp_arg2
                    if exp_arg2[0] == 'ln':
                        return ['^'] + exp_arg2[1:] + exp_arg1
                if exp_operator == '^':
                    if exp_arg1[0] == 'ln':
                        return pn_prettify(['^'] + exp_arg1[1:] + ['^'] + exp_arg1 + ['-'] + exp_arg2 + [1])

        return [operator] + arg1

    return tokens


def __derivative_pn(tokens: List):
    if len(tokens) == 1:
        if is_scalar_pn(tokens):  # if function is scalar
            return [0]
        elif is_pn_single_var(tokens):  # function is linear
            return [1]
        else:
            raise Exception(f'unknown first token: {tokens[0]} \nliteral was expected')
    operator, arg1, arg2 = pn_split_via_operator(tokens)

    if operator in Formula.functions:  # function is function of 1 argument
        functor, argument = operator, arg1

        match functor:
            case 'sin':
                return pn_prettify(['*'] + __derivative_pn(argument) + ['cos'] + argument)
            case 'cos':
                return pn_prettify(['*'] + __derivative_pn(argument) + ['*', -1, 'sin'] + argument)
            case 'sqrt':
                return pn_prettify(['/'] + __derivative_pn(argument) + ['*', 2, 'sqrt'] + argument)
            case 'exp':
                return pn_prettify(['*'] + __derivative_pn(argument) + ['exp'] + argument)
            case 'ln':
                return pn_prettify(['/'] + __derivative_pn(argument) + argument)
            case 'tg':
                return pn_prettify(['/'] + __derivative_pn(argument) + ['^', 'cos'] + argument + [2])
            case 'ctg':
                return pn_prettify(['/', '*', -1] + __derivative_pn(argument) + ['^', 'sin'] + argument + [2])
            case 'arcsin':
                return pn_prettify(['/'] + __derivative_pn(argument) + ['sqrt', '-', 1, '^'] + argument + [2])
            case 'arccos':
                return pn_prettify(['/', '*', -1] + __derivative_pn(argument) + ['sqrt', '-', 1, '^'] + argument + [2])
            case 'arctg':
                return pn_prettify(['/'] + __derivative_pn(argument) + ['+', 1, '^'] + argument + [2])
            case 'arcctg':
                return pn_prettify(['/', '*', -1] + __derivative_pn(argument) + ['+', 1, '^'] + argument + [2])
            case _:
                raise UnableToDifferentiateException(f'cannot find derivative of {functor}')

    elif operator in Formula.operators_precedence:  # function consists of 2 joined by operator
        match operator:
            case '+' | '-':
                # derivative of sum is sum of derivatives
                # derivative of subtraction is subtraction of derivatives
                return pn_prettify([operator] + __derivative_pn(arg1) + __derivative_pn(arg2))

            case '*':
                # (f(x)*g(x))` = f`(x)*g(x)+f(x)*g`(x)
                if is_scalar_pn(arg1):
                    return pn_prettify(['*'] + arg1 + __derivative_pn(arg2))
                if is_scalar_pn(arg2):
                    return pn_prettify(['*'] + arg2 + __derivative_pn(arg1))
                return pn_prettify(['+'] + ['*'] + __derivative_pn(arg1) + arg2 + ['*'] + arg1 + __derivative_pn(arg2))
            case '/':
                # if numerator of fraction is scalar, then treat it like scalar * power function
                # if denominator is scalar, then treat it like scalar * some function
                # in other case, use the quotient rule

                if is_scalar_pn(arg1):
                    return pn_prettify(['*'] + arg1 + __derivative_pn(['^'] + arg2 + [-1]))
                if is_scalar_pn(arg2):
                    return pn_prettify(['/'] + __derivative_pn(arg1) + arg2)
                # else
                return pn_prettify(['/', '-', '*'] + __derivative_pn(arg1) + arg2 +
                                   ['*'] + arg1 + __derivative_pn(arg2) + ['^'] + arg2 + [2])

            case '^':
                # There are 2 cases:
                # 1. exponent is independent of x then it power function x^n -> n*x^(n-1)*(x)`
                # 2. in other case it's  exponential function y^x = e^(ln(y^x)) -> (e^(x*ln(y)))` =
                #    = e^(x*ln(y)) * (x*ln(y))`

                if 'x' not in arg2:
                    return pn_prettify(['*', '*'] + arg2 + ['^'] + arg1 +
                                       ['-'] + arg2 + [1] + __derivative_pn(arg1))
                # else
                return pn_prettify(
                    ['*', 'exp', '*'] + arg2 + ['ln'] + arg1 + __derivative_pn(['*'] + arg2 + ['ln'] + arg1))

    else:
        raise UnableToDifferentiateException(
            f'unknown operator type: {tokens[0]} \nmathematical operator or function was expected')


def derivative(function) -> Formula:
    """
    requires list of pn tokens or Formula object of single variable function on input and then returns
    derivative of it in form of Formula object.
    If it can't calculate derivative, it will raise an exception.
    """
    if not type(function) == list:
        function = function.tokens
    return Formula(__derivative_pn(function))


functions = [
            'x^2 + x^7 + 1/(x^2) + x^(1/3)',
            '(x^7-3x^2+5^x)sin(x)',
            '(5x^3-7x+cos(x))/7^x',
            'sin x + 4 cos x',
            '2ln(x)-3arctg x',
            'x^3*(3ln x -14sqrt(x))',
            'exp(x) * ctg x',
            '(x+1)/(x-3)',
            'exp(x^2+1)',
            'sin(cos(4x))',
            'sin(3x)(2x)/(x^2+1)',
            '(1+14sqrt(x))^11',
            'arctg(2x+3)',
            'x*exp(sqrt(x)+4)/(cos(14x)+4)',
            'sqrt(ln(1/(x-1)))',
            '(cos(3x^5-7x+2))^4',
            'x^(4/3)'
            ]

for function in functions:
    function = Formula(function)
    print(f'function: ${pn_to_latex(function.tokens)}$   \nderivative: ${pn_to_latex(derivative(function).tokens)}$\n')
    print('\\', '='*100, sep='')
    print()


