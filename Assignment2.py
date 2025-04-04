import streamlit as st
import math
import random
import matplotlib.pyplot as plt

def simulate_exponential(lamba, n):
    sample = []
    for i in range(n):
        U = random.random()
        X = (-math.log(U)) / lamba
        sample.append(X)
    return sample

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)

def binomial(n,k):
    return factorial(n) / (factorial(k) * factorial(n-k))

def compute_stationary_dist(f_rate, r_rate, warm, n, k, s):
    '''
    f_rate = failure rate
    r_rate = repair rate
    warm = True for warm stand-by False for cold stand-by
    n = number of components
    k = number of components for the system to work
    s = number of repairman
    '''
    n = int(n)
    k = int(k)
    s = int(s)
    pi = [0.0] * (n + 1)
    if warm:
        for j in range(n+1):
            ratio = f_rate / r_rate
            if j <= s:
                pi[j] = binomial(n, j) * (ratio ** j)
            else:
                # π(j) = [n!/( (n-j)! * s! * s^(j-s) )] * (ratio)^j
                num = factorial(n)
                denom = factorial(n-j) * factorial(s) * (s ** (j-s))
                pi[j] = num / denom * (ratio ** j)
    else:
        pi[0] = 1
        for j in range(1, n+1):
            f_prod = 1
            for i in range(1, j + 1):
                working = n - (i - 1)
                number = min(k, working)
                f_prod *= number * f_rate
            r_prod = 1.0
            for i in range(1, j + 1):
                r_prod *= min(i, s) * r_rate
            pi[j] = pi[0] * (f_prod / r_prod)
    total = sum(pi)
    if total > 0:
        pi = [p / total for p in pi]
    else:
        pi = [0.0] * (n + 1)
    return pi

def compute_availability(f_rate, r_rate, warm, n, k, s):
    pi = compute_stationary_dist(f_rate, r_rate, warm, n, k, s)
    max_failed = n - k
    availability = sum(pi[0:max_failed + 1])
    return availability

def simulate_birth_death(f_rate, r_rate, warm, n, k, s, sim_time = 10000):
    current_state = 0       # Start with j=0 (all components working)
    t = 0.0
    time_in_state = [0.0] * (n + 1)
    states = {}
    while t < sim_time:
        j = current_state ##current number of faulty machines
        if warm:
            lambda_j = (n-j) * f_rate
        else:
            lambda_j = min(n-j, k) * f_rate
        mu_j = min(j, s) * r_rate
        R = mu_j + lambda_j
        if R <= 0: #no transition can occur (j = 0 and lambda_rate = 0 or j = n and mu_rate = 0 -- can't happen but to be sure)
            time_left = sim_time - t
            time_in_state[j] += time_left ##are stuck in this state
            t += time_left ##are stuck in this state
            break
        time_in_state_j = simulate_exponential(R, 1)[0]
        next_event_time = t + time_in_state_j
        if next_event_time > sim_time:
            time_in_state_j = sim_time - t
            next_event_time = sim_time
        time_in_state[j] += time_in_state_j
        states[t] = j
        t = next_event_time
        prob_failure = lambda_j / R
        u = random.random()
        if u < prob_failure:
            if j < n:
                current_state = j + 1 ##another machine failes
        else:
            if j > 0:
                current_state = j - 1
    pi = [t / sim_time for t in time_in_state]
    return pi, states

def plot_birth_death_trajectory(f_rate, r_rate, warm, n, k, s, sim_time=10000):
    """
    Runs the simulation, then creates two plots:
      1) The trajectory of the state j over time (step plot).
      2) A bar chart of the estimated stationary distribution pi(j).
    """
    random.seed(42)
    pi_est, states = simulate_birth_death(f_rate, r_rate, warm, n, k, s, sim_time)

    times_sorted = sorted(states.keys())
    states_sorted = [states[t] for t in times_sorted]

    plt.figure(figsize=(8, 4))
    plt.step(times_sorted, states_sorted, where='post')
    plt.xlabel("Time")
    plt.ylabel("Failed Components (j)")
    plt.title(f"Birth-Death Process Trajectory (n={n}, k={k}, s={s}, warm={warm})")
    plt.grid(True)
    plt.savefig(f"bd_graph_{n}_{k}_{s}_{warm}.png")

    # 2) Plot the estimated stationary distribution as a bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(pi_est)), pi_est)
    plt.xlabel("Number of failed components (j)")
    plt.ylabel("Estimated Probability")
    plt.title("Estimated Stationary Distribution")
    plt.savefig(f"bd_dist_{n}_{k}_{s}_{warm}.png")



def estimate_availability(f_rate, r_rate, warm, n, k, s, sim_time=10000):
    pi, _ = simulate_birth_death(f_rate, r_rate, warm, n, s, k, sim_time)
    avail = sum(pi[0 : (n - k + 1)])
    return avail

def optimization(f_rate, r_rate, warm, componentCost, repairmanCost, downtimeCost, k):
    n_opt = None
    s_opt = None
    minCost = 10e9
    n_opt = None
    s_opt = None
    for n in range(k, k + 11):
        for s in range(1, n + 1):
            avail = estimate_availability(f_rate, r_rate, warm, n, k, s, 10000)
            costOfComponents = componentCost * n
            costOfRepairmen = repairmanCost * s
            costOfDowntime = (1 - avail) * downtimeCost
            totalCost = costOfComponents + costOfRepairmen + costOfDowntime
            if totalCost < minCost:
                minCost = totalCost
                n_opt = n
                s_opt = s
    return (n_opt, s_opt)

def main():
    st.title("Assignment 2")
    st.markdown("""
    This app computes the steady-state availability of a *k-out-of-n* maintenance system and finds the optimal n and s given the failure rate, repair rate, type od standby and 
    
    Choose parameters and see the results.
    """)

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    f_rate = st.sidebar.number_input("Failure rate per component (λ)", min_value=0.0, value=0.1, step=0.01)
    r_rate = st.sidebar.number_input("Repair rate per repairman (μ)", min_value=0.0, value=1.0, step=0.1)
    warm = st.sidebar.selectbox("Warm Standby", [True, False])
    n = st.sidebar.number_input("Total number of components (n)", min_value=1, value=5, step=1)
    k = st.sidebar.number_input("Required working components (k)", min_value=1, max_value=int(n), value=3, step=1)
    s = st.sidebar.number_input("Number of repairmen (s)", min_value=1, max_value=int(n), value=2, step=1)

    k_opt = st.sidebar.number_input("Required working components (k) for optimization", min_value=1, value=3, step=1)
    costComponent = st.sidebar.number_input("Cost of component", min_value = 0.0, value = 1.0, step = 0.5)
    costRepairman = st.sidebar.number_input("Cost of repairman", min_value = 0.0, value = 1.0, step = 0.5)
    costDowntime = st.sidebar.number_input("Cost of downtime", min_value = 0.0, value = 1.0, step = 0.5)


    if k > n:
        st.error("k must be less than or equal to n.")
        st.stop()



    avail = compute_availability(f_rate, r_rate, warm, n, k, s)
    est_avail = estimate_availability(f_rate, r_rate, warm, n, k, s, 100000)
    (n_opt, s_opt) = optimization(f_rate, r_rate, warm, costComponent, costRepairman, costDowntime, int(k_opt))

    st.subheader("Availability Results")

    st.metric("Availability - formulas", f"{avail:.4f}")
    st.metric("Availability - Simulation", f"{est_avail:.4f}")




    st.subheader("Optimization results")
    st.metric("Optimal n:", n_opt)
    st.metric("Optimal s:", s_opt)


    # Display the full stationary distributions for comparison

if __name__ == "__main__":
    main() 




