import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import linalg as LA

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
    
    EPS = np.finfo(np.float64).eps

    numerator = np.inner(weights_act, point_to_be_projected_act) - radius 
    denominator = np.inner(weights_act, weights_act) 
    
    dual = np.divide(numerator, denominator + EPS) # compute the dual variable for the weighted l1-ball projection problem
        
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

    signum = np.sign(point_to_be_projected)
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
        point_to_be_projected_copy_act = np.maximum(x_sol_hyper, 0.0)

        point_to_be_projected_copy[act_ind] = point_to_be_projected_copy_act.copy()
        
        act_ind = point_to_be_projected_copy > 0

        inact_ind_cardinality = sum(x_sol_hyper < 0)
        
        # Check the stopping criteria
        if inact_ind_cardinality == 0:
            x_opt = point_to_be_projected_copy * signum
            break

    # gap = radius -  np.inner(weights, abs(x_opt))
    # print(gap)
    return x_opt, dual

def get_lp_ball_projection(starting_point,
                    point_to_be_projected, 
                                        p,
                                   radius, 
                                  epsilon,
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
    if LA.norm(point_to_be_projected, p) ** p <= radius:  
        return point_to_be_projected
    
    # Step 1 and 2 in IRBP.  
    n = point_to_be_projected.shape[0]
            
    signum = np.sign(point_to_be_projected) 
    yAbs = signum * point_to_be_projected  # yAbs lives in the positive orthant of R^n
    # print(signum)
    lamb = 0.0
    residual_alpha0 = (1. / n) * LA.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
    residual_beta0 =  abs(LA.norm(starting_point, p) ** p - radius)
    
    cnt = 0
    
    # The loop of IRBP
    while True:
            
        cnt += 1
        alpha_res = (1. / n) * LA.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
        beta_res = abs(LA.norm(starting_point, p) ** p - radius)
        
        if max(alpha_res, beta_res) < tol * max(max(residual_alpha0, residual_beta0),\
                                                              1.0) or cnt > MAX_ITER:
            x_final = signum * starting_point # symmetric property of lp ball
            break
        
            
        # Step 3 in IRBP. Compute the weights
        weights = p * 1. / ((np.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)
        
        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = radius - LA.norm(abs(starting_point) + epsilon, p) ** p + np.inner(weights, np.abs(starting_point))
            
        assert gamma_k > 0, "The current Gamma is non-positive"
         
        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = get_weightedl1_ball_projection(yAbs, weights, gamma_k)
        x_new[np.isnan(x_new)] = np.zeros_like(x_new[np.isnan(x_new)])
            
        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = LA.norm(x_new - starting_point, 2) * LA.norm(np.sign(x_new - starting_point) * weights, 2) ** Tau
            
        if condition_left <= condition_right:
            theta = np.minimum(beta_res, 1. / np.sqrt(cnt)) ** (1. / p)
            epsilon = theta * epsilon
            
        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()
    return  x_final

p = 0.999
x = np.random.rand(100)-0.5
y = get_lp_ball_projection(np.zeros_like(x),x,p,10,1e-4)
a_lp = p*np.sign(y)*(abs(y)+1e-15)**(p-1)
B_lp = -np.eye(len(x))
H_lp = np.eye(len(x)) - np.diag(np.sign(y)*(y-x)*(p-1)*(np.abs(y)+1e-15)**-1)
H_inv = np.linalg.inv(H_lp)
a_lp = np.mat(a_lp).T
B_lp = np.mat(B_lp)
H_lp = np.mat(H_lp)

dyx_lp = ((H_inv @ a_lp @ a_lp.T @ H_inv)/(a_lp.T @ H_inv @ a_lp)-H_inv) @ B_lp


z = 10
xx = x.copy()
x = np.abs(x)
# 1. Sort v into mu (decreasing)
mu = np.sort(x)[::-1]
# 2. Find rho (number of strictly positive elements of optimal solution w)
mu_cumulative_sum = mu.cumsum(axis = -1)
rho = np.sum(mu * np.arange(1, len(x) + 1) > (mu_cumulative_sum - z), axis = -1)
# 3. Compute the Lagrange multiplier theta associated with the simplex constraint
#print(mu_cumulative_sum)
theta = (mu_cumulative_sum[(rho - 1)]-z)/ rho
#theta = (torch.gather(mu_cumulative_sum, -1, (rho - 1)) - x) / rho
#print(theta)
# 4. Compute projection
w = (x - theta).clip(min = 0.0)

# 3. Correct the element signs
w = w* np.where(xx < 0, -np.ones_like(xx), np.ones_like(xx))

DYh = np.sign(w)
grad_output = np.random.rand(100)-0.5
grad_input = np.abs(DYh)* grad_output  - DYh * (
            (DYh* grad_output ).sum(axis = -1) / (DYh * DYh).sum(axis = -1))
print(grad_input)
print(grad_output*dyx_lp)
print(grad_input - grad_output*dyx_lp)
print(LA.norm(y-w))




# import torch


# def get_hyperplane_projection(point_to_be_projected_act, 
#                                             weights_act, 
#                                                 radius):
#     """Gets the hyperplane projection of a given point.
    
#     Args:
#         point_to_be_projected_act: Point to be projected with positive components.
#         weights: the weights vector with positive components
#         radius: The radius of weighted l1-ball.
#     Returns:
#         x_sub : The projection point
    
#     """
    
#     EPS = torch.finfo(torch.float64).eps

#     numerator = torch.inner(weights_act, point_to_be_projected_act) - radius 
#     denominator = torch.inner(weights_act, weights_act) 
    
#     dual = torch.divide(numerator, denominator + EPS) # compute the dual variable for the weighted l1-ball projection problem
        
#     x_sub = point_to_be_projected_act - dual * weights_act

#     return x_sub,dual


# def get_weightedl1_ball_projection(point_to_be_projected,
#                                    weights, 
#                                    radius):
#     """Gets the weighted l1 ball projection of given point.
    
#     Args:
#         point_to_be_projected: Point to be projected.
#         weights: the weights vector.
#         radius: The radius of weighted l1-ball.
#     Returns:
#         x_opt : The projection point.
    
#     """

#     signum = torch.sign(point_to_be_projected)
#     point_to_be_projected_copy = signum * point_to_be_projected
    
    
#     act_ind = [True] * point_to_be_projected.shape[0]
    
#     # The loop of the weight l1-ball projection algorithm
#     while True:
#         # Discarding the zeros 
#         point_to_be_projected_copy_act = point_to_be_projected_copy[act_ind]
#         weights_act = weights[act_ind]
        
#         # Perform projections in a reduced space R^{|act_ind|}
#         x_sol_hyper, dual = get_hyperplane_projection(point_to_be_projected_copy_act, weights_act, radius)
        
#         # Update the active index set
#         point_to_be_projected_copy_act = torch.maximum(x_sol_hyper, torch.zeros_like(x_sol_hyper))

#         point_to_be_projected_copy[act_ind] = point_to_be_projected_copy_act.clone()
        
#         act_ind = point_to_be_projected_copy > 0

#         inact_ind_cardinality = sum(x_sol_hyper < 0)
        
#         # Check the stopping criteria
#         if inact_ind_cardinality == 0:
#             x_opt = point_to_be_projected_copy * signum
#             break

#     # gap = radius -  torch.inner(weights, abs(x_opt))
#     # print(gap)
#     return x_opt, dual

# def get_lp_ball_projection(starting_point,
#                     point_to_be_projected, 
#                                         p,
#                                    radius, 
#                                   epsilon=1e-4,
#                                   Tau = 1.1,
#                                   condition_right=100,
#                                   tol=1e-4,
#                                   MAX_ITER=500,**kwargs):
#     """Gets the lp ball projection of given point.

#     Args:
#     ----------
#     point_to_be_projected: Point to be projected.
#     starting_point: Iterates of IRBP.
#     p: p parameter for lp-ball.
#     radius: The radius of lp-ball.
#     epsilon: Initial value of the smoothing parameter epsilon
#     Tau, condition_right: hyperparameters
#     Returns
#     -------
#     x_final : The projection point 
#     dual : The multiplier
#     Flag_gamma_pos : whether IRBP successfully returens a solution
#     count : The number of iterations

#     """
#     if torch.norm(point_to_be_projected, p) ** p <= radius:  
#         return point_to_be_projected
    
#     # Step 1 and 2 in IRBP.  
#     n = point_to_be_projected.shape[0]
            
#     signum = torch.sign(point_to_be_projected) 
#     # print(signum)
#     yAbs = signum * point_to_be_projected  # yAbs lives in the positive orthant of R^n
    
#     lamb = 0.0
#     residual_alpha0 = (1. / n) * torch.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
#     residual_beta0 =  abs(torch.norm(starting_point, p) ** p - radius)
    
#     cnt = 0
    
#     # The loop of IRBP
#     while True:
            
#         cnt += 1
#         alpha_res = (1. / n) * torch.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
#         beta_res = abs(torch.norm(starting_point, p) ** p - radius)
        
#         if max(alpha_res, beta_res) < tol * max(max(residual_alpha0, residual_beta0),\
#                                                               1.0) or cnt > MAX_ITER:
#             x_final = signum * starting_point # symmetric property of lp ball
#             break
        
            
#         # Step 3 in IRBP. Compute the weights
#         weights = p * 1. / ((torch.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)
        
#         # Step 4 in IRBP. Solve the subproblem for x^{k+1}
#         gamma_k = radius - torch.norm(abs(starting_point) + epsilon, p) ** p + torch.inner(weights, torch.abs(starting_point))
            
#         assert gamma_k > 0, "The current Gamma is non-positive"
         
#         # Subproblem solver : The projection onto weighted l1-ball
#         x_new, lamb = get_weightedl1_ball_projection(yAbs, weights, gamma_k)
#         x_new[torch.isnan(x_new)] = torch.zeros_like(x_new[torch.isnan(x_new)])
            
#         # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
#         condition_left = torch.norm(x_new - starting_point, 2) * torch.norm(torch.sign(x_new - starting_point) * weights, 2) ** Tau
            
#         if condition_left <= condition_right:
#             theta = min(beta_res, 1. / np.sqrt(cnt)) ** (1. / p)
#             epsilon = theta * epsilon
            
#         # Step 6 in IRBP. Set k <--- (k+1)
#         starting_point = x_new.clone()
#     return  x_final

# print(grad_output*dyx_lp)
# grad_output = torch.from_numpy(grad_output) 
# x = torch.from_numpy(xx)
# y = get_lp_ball_projection(torch.zeros_like(x),x,p,10,1e-4)
# a_lp = p*torch.sign(y)*(abs(y)+1e-15)**(p-1)
# a_lp = a_lp.unsqueeze(1)
# B_lp = -torch.eye(len(x))
# B_lp = torch.tensor(B_lp, dtype=a_lp.dtype)
# H_inv = torch.diag_embed((1-torch.sign(y)*(y-x)*(p-1)*(torch.abs(y)+1e-15)**-1)**-1) 
# dyx_lp = ((H_inv @ a_lp @ a_lp.T @ H_inv)/(a_lp.T @ H_inv @ a_lp)-H_inv) @ B_lp
# print(dyx_lp@grad_output)