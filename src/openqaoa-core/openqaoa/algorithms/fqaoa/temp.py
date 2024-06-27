# The simplest FQAOA workflow
import networkx
from openqaoa.problems import MinimumVertexCover

g = networkx.circulant_graph(6, [1])
vc = MinimumVertexCover(g, field=1.0, penalty=10)
qubo_problem = vc.qubo

from openqaoa import FQAOA
q = FQAOA()
q.compile(problem=qubo_problem)
#q.compile(qubo_problem)
q.optimize() # L372
# L385: self.result = self.optimizer.qaoa_result
# 
q.result.optimized 
