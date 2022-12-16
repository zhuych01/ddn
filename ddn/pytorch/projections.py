#
# Euclidean projection onto the Lp-sphere
#
# y(x) = argmin_u f(x, u)
# subject to h(u) = 0
#
# where f(x, u) = 0.5 ||u - x||_2^2
#       h(u) = ||u||_p = 1
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


def get_hyperplane_projection(point_to_be_projected_act, 
                                            weights_act, 
                                                radius):
    """Gets the hyperplane projection of a given point.
    
    Args:
        point_to_be_projected_act: Point to be projected with positive components.
        weights: the weights vector with positive components
        radius: The radius of weighted l1-ball.
    Returns:
        x_sub : The projection point
    
    """
    
    EPS = torch.finfo(torch.float64).eps

    numerator = torch.inner(weights_act, point_to_be_projected_act) - radius 
    denominator = torch.inner(weights_act, weights_act) 
    
    dual = torch.divide(numerator, denominator + EPS) # compute the dual variable for the weighted l1-ball projection problem
        
    x_sub = point_to_be_projected_act - dual * weights_act

    return x_sub,dual


def get_weightedl1_ball_projection(point_to_be_projected,
                                   weights, 
                                   radius):
    """Gets the weighted l1 ball projection of given point.
    
    Args:
        point_to_be_projected: Point to be projected.
        weights: the weights vector.
        radius: The radius of weighted l1-ball.
    Returns:
        x_opt : The projection point.
    
    """

    signum = torch.sign(point_to_be_projected)
    point_to_be_projected_copy = signum * point_to_be_projected
    
    
    act_ind = [True] * point_to_be_projected.shape[0]
    
    # The loop of the weight l1-ball projection algorithm
    while True:
        # Discarding the zeros 
        point_to_be_projected_copy_act = point_to_be_projected_copy[act_ind]
        weights_act = weights[act_ind]
        
        # Perform projections in a reduced space R^{|act_ind|}
        x_sol_hyper, dual = get_hyperplane_projection(point_to_be_projected_copy_act, weights_act, radius)
        
        # Update the active index set
        point_to_be_projected_copy_act = torch.maximum(x_sol_hyper, 0.0)

        point_to_be_projected_copy[act_ind] = point_to_be_projected_copy_act.copy()
        
        act_ind = point_to_be_projected_copy > 0

        inact_ind_cardinality = sum(x_sol_hyper < 0)
        
        # Check the stopping criteria
        if inact_ind_cardinality == 0:
            x_opt = point_to_be_projected_copy * signum
            break

    # gap = radius -  torch.inner(weights, abs(x_opt))
    # print(gap)
    return x_opt, dual

def get_lp_ball_projection(starting_point,
                    point_to_be_projected, 
                                        p,
                                   radius, 
                                  epsilon = 1e-4,
                                  Tau = 1.1,
                                  condition_right=100,
                                  tol=1e-8,
                                  MAX_ITER=1000,**kwargs):
    """Gets the lp ball projection of given point.

    Args:
    ----------
    point_to_be_projected: Point to be projected.
    starting_point: Iterates of IRBP.
    p: p parameter for lp-ball.
    radius: The radius of lp-ball.
    epsilon: Initial value of the smoothing parameter epsilon
    Tau, condition_right: hyperparameters
    Returns
    -------
    x_final : The projection point 
    dual : The multiplier
    Flag_gamma_pos : whether IRBP successfully returens a solution
    count : The number of iterations

    """
    if torch.norm(point_to_be_projected, p) ** p <= radius:  
        return point_to_be_projected
    
    # Step 1 and 2 in IRBP.  
    n = point_to_be_projected.shape[0]
            
    signum = torch.sign(point_to_be_projected) 
    yAbs = signum * point_to_be_projected  # yAbs lives in the positive orthant of R^n
    
    lamb = 0.0
    residual_alpha0 = (1. / n) * torch.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
    residual_beta0 =  abs(torch.norm(starting_point, p) ** p - radius)
    
    cnt = 0
    
    # The loop of IRBP
    while True:
            
        cnt += 1
        alpha_res = (1. / n) * torch.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
        beta_res = abs(torch.norm(starting_point, p) ** p - radius)
        
        if max(alpha_res, beta_res) < tol * max(max(residual_alpha0, residual_beta0),\
                                                              1.0) or cnt > MAX_ITER:
            x_final = signum * starting_point # symmetric property of lp ball
            break
        
            
        # Step 3 in IRBP. Compute the weights
        weights = p * 1. / ((torch.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)
        
        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = radius - torch.norm(abs(starting_point) + epsilon, p) ** p + torch.inner(weights, torch.abs(starting_point))
            
        assert gamma_k > 0, "The current Gamma is non-positive"
         
        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = get_weightedl1_ball_projection(yAbs, weights, gamma_k)
        x_new[torch.isnan(x_new)] = torch.zeros_like(x_new[torch.isnan(x_new)])
            
        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = torch.norm(x_new - starting_point, 2) * torch.norm(torch.sign(x_new - starting_point) * weights, 2) ** Tau
            
        if condition_left <= condition_right:
            theta = torch.minimum(beta_res, 1. / torch.sqrt(cnt)) ** (1. / p)
            epsilon = theta * epsilon
            
        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()
    return  x_final


class Simplex():
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto a positive simplex

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to sum_i w_i = z, w_i >= 0 

        using the algorithm (Figure 1) from:
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions,
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra,
            International Conference on Machine Learning (ICML 2008),
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the simplex

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the simplex

        Complexity:
            O(n log(n))
            A linear time alternative is proposed in [1], similar to using a
            selection algorithm instead of sorting.
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Sort v into mu (decreasing)
        mu, _ = v.sort(dim = -1, descending = True)
        # 2. Find rho (number of strictly positive elements of optimal solution w)
        mu_cumulative_sum = mu.cumsum(dim = -1)
        rho = torch.sum(mu * torch.arange(1, v.size()[-1] + 1, dtype=v.dtype, device=v.device) > (mu_cumulative_sum - z), dim = -1, keepdim=True)
        # 3. Compute the Lagrange multiplier theta associated with the simplex constraint
        theta = (torch.gather(mu_cumulative_sum, -1, (rho - 1)) - z) / rho.type(v.dtype)
        # 4. Compute projection
        w = (v - theta).clamp(min = 0.0)
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        # Select active positivity constraints
        mask = torch.where(output > 0.0, torch.ones_like(input), torch.zeros_like(input))
        masked_output = mask * grad_output
        grad_input = masked_output - mask * (
            masked_output.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L1Sphere(Simplex):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L1-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_1 = z

        using the algorithm (Figure 1) from:
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions,
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra,
            International Conference on Machine Learning (ICML 2008),
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L1-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L1-sphere

        Complexity:
            O(n log(n))
            A linear time alternative is proposed in [1], similar to using a
            selection algorithm instead of sorting.
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # # 1. Replace v = 0 with v = [1, 0, ..., 0]
        # mask = torch.isclose(v, torch.zeros_like(v), rtol=0.0, atol=1e-12).sum(dim=-1, keepdim=True) == v.size(-1)
        # unit_vector = F.one_hot(v.new_zeros(1, dtype=torch.long), num_classes=v.size(-1)).type(v.dtype)
        # v = torch.where(mask, unit_vector, v)
        # 1. Take the absolute value of v
        u = v.abs()
        # 2. Project u onto the positive simplex
        beta, _ = Simplex.project(u, z=z)
        # 3. Correct the element signs
        w = beta * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        DYh = output.sign()
        grad_input = DYh.abs() * grad_output - DYh * (
            (DYh * grad_output).sum(-1, keepdim=True) / (DYh * DYh).sum(-1, keepdim=True))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L1Ball(L1Sphere):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L1-ball

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_1 <= z

        using the algorithm (Figure 1) from:
        [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions,
            John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra,
            International Conference on Machine Learning (ICML 2008),
            http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L1-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L1-ball

        Complexity:
            O(n log(n))
            A linear time alternative is proposed in [1], similar to using a
            selection algorithm instead of sorting.
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Project onto L1 sphere
        w, _ = L1Sphere.project(v, z=z)
        # 2. Select v if already inside ball, otherwise select w
        is_outside = v.abs().sum(dim=-1, keepdim=True).gt(z)
        w = torch.where(is_outside, w, v)
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Compute constrained gradient
        grad_input = L1Sphere.gradient(grad_output, output, input, is_outside)
        # 2. If input was already inside ball (or on surface), use unconstrained gradient instead
        grad_input = torch.where(is_outside, grad_input, grad_output)
        return grad_input

class L2Sphere():
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L2-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_2 = z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L2-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L2-sphere

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # Replace v = 0 with unit vector:
        mask = torch.isclose(v, torch.zeros_like(v), rtol=0.0, atol=1e-12).sum(dim=-1, keepdim=True) == v.size(-1)
        unit_vector = torch.ones_like(v).div(torch.ones_like(v).norm(p=2, dim=-1, keepdim=True))
        v = torch.where(mask, unit_vector, v)
        # Compute projection:
        w = z * v.div(v.norm(p=2, dim=-1, keepdim=True))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # ToDo: Check for div by zero
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        input = input.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        output_norm = output.norm(p=2, dim=-1, keepdim=True)
        input_norm = input.norm(p=2, dim=-1, keepdim=True)
        ratio = output_norm.div(input_norm)
        grad_input = ratio * (grad_output - output * (
            output * grad_output).sum(-1, keepdim=True).div(output_norm.pow(2)))
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class L2Ball(L2Sphere):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an L2-ball

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_2 <= z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the L2-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the L2-ball

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Project onto L2 sphere
        w, _ = L2Sphere.project(v, z=z)
        # 2. Select v if already inside ball, otherwise select w
        is_outside = v.norm(p=2, dim=-1, keepdim=True).gt(z)
        w = torch.where(is_outside, w, v)
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Compute constrained gradient
        grad_input = L2Sphere.gradient(grad_output, output, input, is_outside)
        # 2. If input was already inside ball (or on surface), use unconstrained gradient instead
        grad_input = torch.where(is_outside, grad_input, grad_output)
        return grad_input

class LInfSphere():
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an LInf-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_infinity = z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the LInf-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the LInf-sphere

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # 1. Take the absolute value of v
        u = v.abs()
        # 2. Project u onto the (non-negative) LInf-sphere
        # If u_i >= z, u_i = z
        # If u_i < z forall i, find max and set to z
        z = torch.tensor(z, dtype=v.dtype, device=v.device)
        u = torch.where(u.gt(z), z, u)
        u = torch.where(u.ge(u.max(dim=-1, keepdim=True)[0]), z, u)
        # 3. Correct the element signs
        w = u * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, None

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Flatten:
        output_size = output.size()
        output = output.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)
        # 2. Use implicit differentiation to compute derivative
        mask = output.abs().ge(output.abs().max(dim=-1, keepdim=True)[0])
        hY = output.sign() * mask.type(output.dtype)
        grad_input = grad_output - hY.abs() * grad_output
        # 3. Unflatten:
        grad_input = grad_input.reshape(output_size)
        return grad_input

class LInfBall(LInfSphere):
    @staticmethod
    def project(v, z = 1.0):
        """ Euclidean projection of a batch of vectors onto an LInf-ball

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_infinity <= z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the LInf-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the LInf-ball

        Complexity:
            O(n)
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        # Using LInfSphere.project is more expensive here
        # 1. Take the absolute value of v
        u = v.abs()
        is_outside = u.max(dim=-1, keepdim=True)[0].gt(z) # Store for backward pass
        # 2. Project u onto the (non-negative) LInf-sphere if outside
        # If u_i >= z, u_i = z
        z = torch.tensor(z, dtype=v.dtype, device=v.device)
        u = torch.where(u.gt(z), z, u)
        # 3. Correct the element signs
        w = u * torch.where(v < 0, -torch.ones_like(v), torch.ones_like(v))
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # 1. Compute constrained gradient
        grad_input = LInfSphere.gradient(grad_output, output, input, is_outside)
        # 2. If input was already inside ball (or on surface), use unconstrained gradient instead
        grad_input = torch.where(is_outside, grad_input, grad_output)
        return grad_input

class EuclideanProjectionFn(torch.autograd.Function):
    """
    A function to project a set of features to an Lp-sphere or Lp-ball
    """
    @staticmethod
    def forward(ctx, input, method, radius):
        output, is_outside = method.project(input, radius.item())
        ctx.method = method
        ctx.save_for_backward(output.clone(), input.clone(), is_outside)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input, is_outside = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.method.gradient(grad_output, output, input, is_outside)
        return grad_input, None, None

class EuclideanProjection(torch.nn.Module):
    def __init__(self, method, radius = 1.0):
        super(EuclideanProjection, self).__init__()
        self.method = method
        self.register_buffer('radius', torch.tensor([radius]))

    def forward(self, input):
        return EuclideanProjectionFn.apply(input,
                                           self.method,
                                           self.radius
                                           )

    def extra_repr(self):
        return 'method={}, radius={}'.format(
            self.method.__name__, self.radius
        )


class LpBall():
    @staticmethod
    def project(v, z = 1.0,p = 0.8):
        """ Euclidean projection of a batch of vectors onto an L2-sphere

        Solves:
            minimise_w 0.5 * || w - v ||_2^2
            subject to ||w||_p = z

        Arguments:
            v: (..., n) Torch tensor,
                batch of n-dimensional vectors to project

            z: float, optional, default: 1.0,
                radius of the Lp-ball

        Return Values:
            w: (..., n) Torch tensor,
                Euclidean projection of v onto the Lp-sphere

        Convergence rate:
            O(1/sqrt(k))
        """
        assert z > 0.0, "z must be strictly positive (%f <= 0)" % z
        assert p > 0.0 and p < 1.0, "p should between 0.0 and 1.0"
        # Replace v = 0 with unit vector:
        w, _ = get_lp_ball_projection(torch.zeros_like(v),v,p,z)
        # 2. Select v if already inside ball, otherwise select w
        is_outside = v.norm(p=p, dim=-1, keepdim=True)**p<=z
        w = torch.where(is_outside, w, v)
        return w, is_outside

    @staticmethod
    def gradient(grad_output, output, input, is_outside = None):
        # Compute vector-Jacobian product (grad_output * Dy(x))
        # NEED TO BE DONE
        pass

""" Check gradients
from torch.autograd import gradcheck

# method = Simplex
method = L1Sphere
# method = L1Ball
# method = L2Sphere
# method = L2Ball
# method = LInfSphere
# method = LInfBall

radius = 100.0
radius = 1.0
# radius = 0.5

projection = EuclideanProjectionFn.apply
radius_tensor = torch.tensor([radius], requires_grad=False)
features = torch.randn(4, 2, 2, 100, dtype=torch.double, requires_grad=True)
input = (features, method, radius_tensor)
test = gradcheck(projection, input, eps=1e-6, atol=1e-4)
print("{}: {}".format(method.__name__, test))

# Check projections
features = torch.randn(1, 1, 1, 10, dtype=torch.double, requires_grad=True)
input = (features, method, radius_tensor)
print(features.sum(dim=-1))
print(features.abs().sum(dim=-1))
print(features.norm(p=2, dim=-1))
print(features.abs().max(dim=-1)[0])
print(features)
output = projection(*input)
print(output.sum(dim=-1))
print(output.abs().sum(dim=-1))
print(output.norm(p=2, dim=-1))
print(output.abs().max(dim=-1)[0])
print(output)
"""