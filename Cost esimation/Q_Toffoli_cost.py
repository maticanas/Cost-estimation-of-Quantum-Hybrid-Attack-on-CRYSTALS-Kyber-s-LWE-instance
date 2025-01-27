import math

# Length Prameters

def compute_l_r(r):
    """Computes l_r = ceil(log2(r + 1))."""
    return ceil_log2(r + 1)

def compute_k(n, r):
    """Computes k = 2n - r."""
    return 2 * n - r

def compute_l_k(k):
    return ceil_log2(k)

def compute_l_t(l_r):
    """Computes l_t = 14 + l_r."""
    return 14 + l_r

def compute_l_tr(l_r):
    """Computes l_tr = 26 + l_r."""
    return 26 + l_r

def compute_l_p(l_t, f, r, k):
    """Computes l_p = l_t + ceil(fr + 2log2(k))."""
    return l_t + ceil(f * r  + 2 * math.log2(k))

def compute_l_tilde_b(l_p):
    """Computes l_tilde_b = 12 + l_p."""
    return 12 + l_p

def compute_l_t_tilde_b(l_t, l_tilde_b):
    """Computes l_t_tilde_b = l_t + l_tilde_b."""
    return l_t + l_tilde_b

def compute_l_M(l_t_tilde_b, l_k):
    """Computes l_M = l_t_tilde_b + l_k."""
    return l_t_tilde_b + l_k

def compute_l_u_j(l_t, l_q, l_k, log2_L_j):
    """Computes l_{u_j} = l_M - l_p - floor(log2(L_j))."""
    return l_t + l_q + l_k - math.floor(log2_L_j)

# def compute_l_u_jb_j_x(l_u_j, l_b_j_x):
#     """Computes l_{u_jb_{j,x}} = l_u_j + l_b_{j,x}."""
#     return l_u_j + l_b_j_x

# def compute_l_u_jb_j(k, l_u_jb_j_x):
#     """Computes l_{u_jb_j} = k * l_{u_jb_{j,x}}."""
#     return k * l_u_jb_j_x

def compute_l_L_j(l_p, log2_L_j):
    """Computes l_{L_j} = ceil(log2(L_j)) + l_p."""
    return ceil(log2_L_j) + l_p



# Quantum Arithmetic Costs

def ceil(x):
    """Computes ceil(x)."""
    return math.ceil(x)

def ceil_log2(x):
    """Computes ceil(log2(x))."""
    return ceil(math.log2(x))

def table_lookup_cost(w, n):
    """Toffoli cost for Table Lookup."""
    return 2**(w + 1) - 4

def addition_cost(n):
    """Toffoli cost for Addition."""
    return 2 * n - 2

def constant_addition_cost(n):
    """Toffoli cost for Constant Addition."""
    return 2 * n - 4

def controlled_constant_addition_cost(n):
    """Toffoli cost for Controlled Constant Addition."""
    return 2 * n - 2

def modular_addition_cost(n):
    """Toffoli cost for Modular Addition."""
    return 8 * n - 4

def constant_modular_addition_cost(n):
    """Toffoli cost for Constant Modular Addition."""
    return 8 * n - 6

def unsigned_product_addition_cost(n, m):
    """Toffoli cost for Unsigned Product Addition."""
    w = ceil_log2(n)
    return ceil(n / w) * (n + 2 * m + 2 ** (w + 2) - 9) - (n + 2 * m - 1)

def positive_constant_multiplication_cost(n, m):
    """Toffoli cost for Positive Constant Multiplication."""
    pa_c_cost = unsigned_product_addition_cost(n - 1, m)
    return pa_c_cost + 2 * (n + m) - 2

def negative_constant_multiplication_cost(n, m):
    """Toffoli cost for Negative Constant Multiplication."""
    pa_c_cost = unsigned_product_addition_cost(n - 1, m)
    return pa_c_cost + 4 * (n + m) - 4

def unsigned_modular_product_addition_cost(n, m):
    """Toffoli cost for Unsigned Modular Product Addition."""
    w = ceil_log2(n)
    return ceil(n / w) * (2**(w + 2) + 8 * n - 12) - (8 * n - 4)

def constant_modular_multiplication_cost(n, m):
    """Toffoli cost for Constant Modular Multiplication."""
    pa_c_q_cost = unsigned_modular_product_addition_cost(n, m)
    return pa_c_q_cost

def constant_division_cost(n, m):
    """Toffoli cost for Constant Division."""
    return (4 * m - 2) * n - 2 * m**2 - 4 * m + 3


# Component Circuit Costs

def compute_cw_g_toffoli_depth(k, l_t, geta_1):
    """Computes Toffoli depth for Cw_g circuit."""
    return k * (addition_cost(l_t) + geta_1)

def compute_cw_g_toffoli_cost(k, l_t, geta_1, r):
    """Computes Toffoli cost for Cw_g circuit."""
    return k*(r+1)*addition_cost(l_t) + geta_1*k*r


def Mj_toffoli_depth(l_t, l_tilde_b, l_t_tilde_b, l_k):
    sum_depth = 0
    for i in range(1, l_k + 1):
        sum_depth += addition_cost(l_t_tilde_b + i)
    return 2 * negative_constant_multiplication_cost(l_t, l_tilde_b) + sum_depth

def Mj_toffoli_cost(l_t, l_tilde_b, l_t_tilde_b, k):
    sum_cost = 0
    for i in range(1, k + 1):
        coeff = ceil(k/(2**i))
        sum_cost += coeff * addition_cost(l_t_tilde_b + i)
    return 2 * k * negative_constant_multiplication_cost(l_t, l_tilde_b) + sum_cost


def compute_u_j_toffoli_depth(l_M, l_L_j, l_u_j):
    """Computes Toffoli depth for u_j circuit."""
    return 2 * constant_division_cost(l_M + 1, l_L_j) + 2 * constant_addition_cost(l_u_j)

def compute_u_j_toffoli_cost(l_M, l_L_j, l_u_j):
    """Computes Toffoli cost for u_j circuit."""
    return 2 * constant_division_cost(l_M + 1, l_L_j) + 2 * constant_addition_cost(l_u_j)


#need to recheck
def REDj_toffoli_depth(l_u_j, l_b_j_x, l_tr):
    return 2 * negative_constant_multiplication_cost(l_u_j, l_b_j_x) + addition_cost(l_tr)

#need to recheck
def REDj_toffoli_cost(l_u_j, l_b_j_x, l_tr, k):
    return 2 * k * negative_constant_multiplication_cost(l_u_j, l_b_j_x) + k * addition_cost(l_tr)

def rangecheck_toffoli_depth(l_tr, k):
    return 2 * addition_cost(l_tr) + 2 * ceil_log2(l_tr - 3) + 2 * ceil_log2(k)

def rangecheck_toffoli_cost(l_tr, k):
    return 2 * k * addition_cost(l_tr) + 2 * k * l_tr - 4 * k - 3


def compute_lwecheck_toffoli_depth(m, n, l_q, geta_1):
    """Computes Toffoli depth for LWECheck circuit."""
    return (m + 1) * modular_addition_cost(l_q) + constant_modular_addition_cost(l_q) + 2 * ceil_log2(m * l_q) + geta_1 * m - 1

def compute_lwecheck_toffoli_cost(m, n, l_q, geta_1):
    """Computes Toffoli cost for LWECheck circuit."""
    return (m*n + m) * modular_addition_cost(l_q) + m*constant_modular_addition_cost(l_q) + 2 * m * l_q + geta_1 * m * n - 3



















def compute_all(n, r, log2_L_j_list):
    # Determine f, eta_1, and geta_1

    m = n

    if n == 512:
        f = 1.302
        eta_1 = 2
        geta_1 = 16
    else:
        f = 1.108
        eta_1 = 3
        geta_1 = 20

    # Parameters
    l_q = 12
    l_b_j_x = 12
    l_r = compute_l_r(r)
    k = compute_k(n, r)
    l_t = compute_l_t(l_r)
    l_tr = compute_l_tr(l_r)
    l_p = compute_l_p(l_t, f, r, k)
    l_tilde_b = compute_l_tilde_b(l_p)
    l_t_tilde_b = compute_l_t_tilde_b(l_t, l_tilde_b)
    l_k = compute_l_k(k)
    l_M = compute_l_M(l_t_tilde_b, l_k)

    rangecheck_cost = rangecheck_toffoli_cost(l_tr, k)
    rangecheck_depth = rangecheck_toffoli_depth(l_tr, k)
    
    mj_cost = Mj_toffoli_cost(l_t, l_tilde_b, l_t_tilde_b, k)
    mj_depth = Mj_toffoli_depth(l_t, l_tilde_b, l_t_tilde_b, l_k)

    lwecheck_cost = compute_lwecheck_toffoli_cost(m, n, l_q, geta_1)
    lwecheck_depth = compute_lwecheck_toffoli_depth(m, n, l_q, geta_1)

    cw_g_cost = compute_cw_g_toffoli_cost(k, l_t, geta_1, r)
    cw_g_depth = compute_cw_g_toffoli_depth(k, l_t, geta_1)

    NP_B_depth = 0
    NP_B_cost = 0

    for j in range(k):
        log2_L_j = log2_L_j_list[j] 
        l_u_j = compute_l_u_j(l_t, l_q, l_k, log2_L_j)
        l_L_j = compute_l_L_j(l_p, log2_L_j)

        u_j_cost = compute_u_j_toffoli_cost(l_M, l_L_j, l_u_j)
        u_j_depth = compute_u_j_toffoli_depth(l_M, l_L_j, l_u_j)

        redj_cost = REDj_toffoli_cost(l_u_j, l_b_j_x, l_tr, k)
        redj_depth = REDj_toffoli_depth(l_u_j, l_b_j_x, l_tr)

        NP_B_j_depth = u_j_depth + mj_depth + redj_depth
        NP_B_j_cost = 2 * u_j_cost + mj_cost + redj_cost

        # + TD(u_1)
        if j == 0:
            NP_B_j_depth += u_j_depth

        NP_B_depth += NP_B_j_depth
        NP_B_cost += NP_B_j_cost
    

    S_chi_depth = 2*cw_g_depth + 2 * NP_B_depth + 2*lwecheck_depth + 2*rangecheck_depth + 1
    S_chi_cost = 2*cw_g_cost + 2 * NP_B_cost + 2*lwecheck_cost + 2*rangecheck_cost + 1

    Q_depth = S_chi_depth + 2 * ceil_log2(3*r) - 1
    Q_cost = S_chi_cost + 2*3*r - 3

    return {
        "parameters": {
            "l_q": l_q,
            "l_b_j_x": l_b_j_x,
            "l_r": l_r,
            "k": k,
            "l_t": l_t,
            "l_tr": l_tr,
            "l_p": l_p,
            "l_tilde_b": l_tilde_b,
            "l_t_tilde_b": l_t_tilde_b,
            "l_k": l_k,
            "l_M": l_M,
            "l_u_j": l_u_j,
            "l_L_j": l_L_j,
        },
        "rangecheck": {"cost": rangecheck_cost, "depth": rangecheck_depth},
        "Mj": {"cost": mj_cost, "depth": mj_depth},
        "u_j": {"cost": u_j_cost, "depth": u_j_depth},
        "REDj": {"cost": redj_cost, "depth": redj_depth},
        "LWECheck": {"cost": lwecheck_cost, "depth": lwecheck_depth},
        "Cw_g": {"cost": cw_g_cost, "depth": cw_g_depth},
        "NP_B": {"depth": NP_B_depth, "cost": NP_B_cost},
        "S_chi": {"depth": S_chi_depth, "cost": S_chi_cost},
        "Q": {"depth": Q_depth, "cost": Q_cost},
    }

def compute_Toffoli_depth_and_cost(n, r, log2_L_j_list):

    # Compute parameters and results
    compute_all_results = compute_all(n, r, log2_L_j_list)

    S_chi_depth = compute_all_results["S_chi"]["depth"]
    S_chi_cost = compute_all_results["S_chi"]["cost"]
    Q_depth = compute_all_results["Q"]["depth"]
    Q_cost = compute_all_results["Q"]["cost"]

    # Store results for each n
    result = {
        "n": n,
        "max_S_chi_depth": S_chi_depth,
        "max_S_chi_cost": S_chi_cost,
        "max_Q_depth": Q_depth,
        "max_Q_cost": Q_cost,
        "details": compute_all_results,
    }

    log2_Q_depth = math.log2(Q_depth)
    log2_Q_cost = math.log2(Q_cost)

    return log2_Q_depth, log2_Q_cost



def compute_max_depth_costs():
    results = []
    for n in [512, 768, 1024]:
        max_s_chi_depth = 0
        max_s_chi_cost = 0
        max_q_depth = 0
        max_q_cost = 0
        max_details = {
            "S_chi_depth": {},
            "S_chi_cost": {},
            "Q_depth": {},
            "Q_cost": {},
        }

        print("Computing for n =", n)
        for r in range(1, n):
            for log2_L_j in range(1, 25):
                # Compute parameters and results
                compute_all_results = compute_all(n, r, (2*n-r)*[log2_L_j])

                S_chi_depth = compute_all_results["S_chi"]["depth"]
                S_chi_cost = compute_all_results["S_chi"]["cost"]
                Q_depth = compute_all_results["Q"]["depth"]
                Q_cost = compute_all_results["Q"]["cost"]

                # Update maximum values
                if S_chi_depth > max_s_chi_depth:
                    max_s_chi_depth = S_chi_depth
                    max_details["S_chi_depth"] = {"n": n, "r": r, "log2_L_j": log2_L_j}

                if S_chi_cost > max_s_chi_cost:
                    max_s_chi_cost = S_chi_cost
                    max_details["S_chi_cost"] = {"n": n, "r": r, "log2_L_j": log2_L_j}

                if Q_depth > max_q_depth:
                    max_q_depth = Q_depth
                    max_details["Q_depth"] = {"n": n, "r": r, "log2_L_j": log2_L_j}

                if Q_cost > max_q_cost:
                    max_q_cost = Q_cost
                    max_details["Q_cost"] = {"n": n, "r": r, "log2_L_j": log2_L_j}

        # Store results for each n
        results.append({
            "n": n,
            "max_S_chi_depth": max_s_chi_depth,
            "max_S_chi_cost": max_s_chi_cost,
            "max_Q_depth": max_q_depth,
            "max_Q_cost": max_q_cost,
            "details": max_details,
        })

    # Print results
    for res in results:
        print(f"Results for n = {res['n']}:")
        # print(f"  Max S_chi_depth = {res['max_S_chi_depth']} (log2 scale: {math.log2(res['max_S_chi_depth']):.5f}) "
        #       f"(Details: {res['details']['S_chi_depth']})")
        # print(f"  Max S_chi_cost = {res['max_S_chi_cost']} (log2 scale: {math.log2(res['max_S_chi_cost']):.5f}) "
        #       f"(Details: {res['details']['S_chi_cost']})")
        print(f"  Max Q_depth = {res['max_Q_depth']} (log2 scale: {math.log2(res['max_Q_depth']):.5f}) "
              f"(Details: {res['details']['Q_depth']})")
        print(f"  Max Q_cost = {res['max_Q_cost']} (log2 scale: {math.log2(res['max_Q_cost']):.5f}) "
              f"(Details: {res['details']['Q_cost']})")
        # print log2 scale value



if __name__ == "__main__":
    compute_max_depth_costs()
