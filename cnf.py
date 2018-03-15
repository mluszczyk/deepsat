import random
from typing import List, Tuple, Set
from random import randint, choice

from sympy.logic import simplify_logic
from sympy.logic.inference import satisfiable
from sympy.parsing.sympy_parser import parse_expr


def get_random_kcnf(k, n, m):
    """returns k-CNF with max n variables and m clauses"""
    clauses = []
    for i in range(m):
        clause = []
        for j in range(k):
            var = randint(1, n)
            sign = choice([1, -1])
            svar = sign * var
            clause.append(svar)
        clauses.append(clause)
    return CNF(clauses)


def get_random_kcnfs(sample_number, clause_size, variable_number,
                     clause_number):
    rcnfs = [get_random_kcnf(clause_size, variable_number,
                             random.randint(1, clause_number))
             for _ in range(sample_number)]
    return rcnfs


class CNF(object):
    def __init__(self, clauses: List[List[int]]):
        clauses = tuple(tuple(c) for c in clauses)
        if clauses:
            svars = set(abs(x)
                        for clause in clauses
                        for x in clause)
        else:
            svars = set()

        assert all((isinstance(x, int) and x > 0) for x in svars)

        assert all(all((abs(x) in svars) for x in c) for c in clauses)

        self.vars = svars
        self.clauses = clauses

    def __str__(self):
        def num_to_letter(num):
            assert 1 <= num <= 25
            return chr(num-1+ord('a'))
        tokens = []
        for clause in self.clauses:
            tokens.append('(')
            for var in clause:
                if var < 0:
                    tokens.append('~')
                tokens.append(num_to_letter(abs(var)))
                tokens.append('|')
            tokens.pop()
            tokens.append(')')
            tokens.append('&')
        tokens.pop()
        return ''.join(tokens)

    def to_sympy(self):
        return parse_expr(str(self))

    def simplified(self):
        return simplify_logic(parse_expr(str(self)))

    def satisfiable(self):
        return bool(satisfiable(self.to_sympy()))

    def set_var(self, v):
        av = abs(v)
        assert av in self.vars
        new_clauses = []
        existed = False
        for c in self.clauses:
            if v in c:
                existed = True
                continue
            if -v in c:
                existed = True
                c = set(c)
                c.remove(-v)
            new_clauses.append(c)
        if not existed:
            raise ValueError("Variable didn't exist.")
        return CNF(new_clauses)

    def is_true(self):
        return not self.clauses

    def is_false(self):
        return any(not c for c in self.clauses)

    def __hash__(self):
        return hash(self.clauses)

    def __eq__(self, other):
        return self.clauses == other.clauses


class DPLL(object):
    def __init__(self):
        self.number_of_runs = 0

    def run(self, cnf: CNF):
        assert isinstance(cnf, CNF)
        self.number_of_runs += 1

        if cnf.is_true():
            return []
        elif cnf.is_false():
            return None

        sug_var = self.suggest(cnf)
        sug_cnf = cnf.set_var(sug_var)

        sug_res = self.run(sug_cnf)
        if sug_res is not None:
            return [sug_var] + sug_res

        not_sug_cnf = cnf.set_var(-sug_var)
        not_sug_res = self.run(not_sug_cnf)
        if not_sug_res is not None:
            return [-sug_var] + not_sug_res
        return None

    def suggest(self, cnf):
        return cnf.vars[0]


def main():
    s = CNF([
        [1, 2],
        [-2, 3],
        [-3],
        [3],
    ])
    print(s)
    print(DPLL().run(s))

    rcnf = get_random_kcnf(3, 4, 20)
    print(rcnf)
    print(DPLL().run(rcnf))
    print(rcnf.simplified())


if __name__ == "__main__":
    main()
