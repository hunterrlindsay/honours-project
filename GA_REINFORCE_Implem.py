import time
import numpy as np
import baselines.common.tf_util as U
import tensorflow as tf
import random
from collections import deque
from baselines import logger
from time import gmtime, strftime

POP_SIZE = 1280+1
SELECTED_SIZE = 280
IND_SIZE = 0
GENERATIONS = 1000
EVAL_ITERS = 5
ALPHA = 0
INIT_ALPHA = 0
STEPS_SO_FAR = 0
RAND_MUT_POWER = 0
MAX_STEPS = 150000000
solved_score = 0
action_clip = False
INIT_MUTATE = 0
degrade = False

__author__ = "Hunter Lindsay, Yiming Peng"

def learn(env, policy_fn, *, timesteps_per_actorbatch, max_timesteps = 0,
          max_episodes = 0, max_iters = 0, max_seconds = 0, seed, env_id, params):

    timestr = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    global ALPHA
    global INIT_ALPHA
    global POP_SIZE
    global SELECTED_SIZE
    global action_clip
    global INIT_MUTATE
    global action_clip

    ALPHA = params[0]
    INIT_ALPHA = ALPHA
    POP_SIZE = params[1]
    SELECTED_SIZE = params[2]
    envID = params[3]
    solved_score = params[4]
    action_clip = params[5]
    RAND_MUT_POWER = params[6]
    INIT_MUTATE = RAND_MUT_POWER
    EVAL_ITERS = params[7]
    degrade = params[8]

    logger.log("GAR: "+"Degrade: "+str(degrade)+", Alpha: "+str(ALPHA)+", Pop Size: "+str(POP_SIZE)+", Sel Size: "+str(SELECTED_SIZE)+", "
            "Env: "+envID+", Solved at: "+str(solved_score)+", AC Clip: "+str(action_clip)+", Mutate: "+str(RAND_MUT_POWER)+", Evals: "+str(EVAL_ITERS))

    #Env is the enviroment the player acts in
    ob_space = env.observation_space
    ac_space = env.action_space

    #Policy is our player
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy

    ob = U.get_placeholder_cached(name="ob")
    import numpy as np
    np.random.seed(seed)

    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # traj return - baseline
    ac = pi.pdtype.sample_placeholder([None]) #Action placeholder
    reinforce_loss = tf.reduce_sum(pi.pd.neglogp(ac) * ret) #loss
    var_list = pi.get_trainable_variables()

    global get_gradient
    get_gradient = U.function([ob, ac, ret], [U.flatgrad(reinforce_loss, var_list)])

    U.initialize()

    set_from_flat = U.SetFromFlat(pi.get_trainable_variables())

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    stochastic = False
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    flatten_weights = pi.get_Flat_variables()()

    IND_SIZE = len(flatten_weights)

    eval_iterations = EVAL_ITERS # change to 5 or 3

    best_solution = flatten_weights
    best_fitness = 0

    prev_population = np.ndarray(shape=[POP_SIZE], dtype=np.ndarray)
    np.random.seed(seed)
    population = np.random.randn(POP_SIZE, IND_SIZE)
    prev_fitnesses = np.ndarray(shape=[POP_SIZE])
    fitnesses = np.ndarray(shape=[POP_SIZE])
    timeout = time.time() + 180000  # 5 Days

    for g in range(0, GENERATIONS):
        for i in range(0, POP_SIZE):
            if g == 0:
                results = es_eval(env, env_id, pi,
                                  population[i], best_solution, eval_iterations,
                                  stochastic, timesteps_per_actorbatch, seed, best_fitness, id, set_from_flat)

                # A new ind is produced via the REINFORCE algorithm
                population[i] = reinforce(results, population[i], env)  # Update individual with directed local search based evolution
                fitnesses[i] = results[1]

                logger.log("Ind: " + str(i) + ",  with fitness: " + str(fitnesses[i]) +", Alpha: "+str(ALPHA)+", at time " +
                           strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " generation: " + str(g))

            else:
                if i == 0: # Leaves the first element in place fittest survives
                    population[i] = prev_population[i]
                    fitnesses[i] = prev_fitnesses[i]
                    logger.log("Ind: " + str(i) + ",  with fitness: " + str(fitnesses[i]) + ", at time " +
                               strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " generation: " + str(g))
                else:
                    #Evaluate a given ind for trajectories and fitness
                    parent = prev_population[random.randint(0, SELECTED_SIZE)] + np.random.normal(loc=0, scale=RAND_MUT_POWER, size=IND_SIZE)
                    results = es_eval(env, env_id, pi,
                                      parent, best_solution, eval_iterations,
                                      stochastic, timesteps_per_actorbatch, seed, best_fitness, id, set_from_flat)

                    #A new ind is produced via the REINFORCE algorithm
                    population[i] = reinforce(results, parent, env) #Update individual with directed local search based evolution
                    fitnesses[i] = results[1]

                    logger.log("Ind: " + str(i) + ",  with fitness: " + str(fitnesses[i]) + ", Alpha: " + str(
                        ALPHA) + ", at time " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " generation: " + str(g))

        # Sort this generation by fitness descending order
        sorted_inds = fitnesses.argsort()[::-1]
        fitnesses = fitnesses[sorted_inds]
        population = population[sorted_inds]

        best_fitness = fitnesses[0]
        best_solution = population[0]

        set_from_flat(best_solution)

        if time.time() > timeout:
            logger.log("Ran out of time, best so far:")
            logger.log("Best Fitness: " + str(fitnesses[0]) + ", at time " +
                       strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " generation: " + str(g))
            break

        if best_fitness >= solved_score:
            logger.log("Best Fitness: " + str(fitnesses[0]) + ", at time " +
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " generation: " + str(g))
            break

        else:
            logger.log("Best Fitness: "+str(fitnesses[0])+", at time "+
                       strftime("%Y-%m-%d %H:%M:%S", gmtime())+" generation: "+str(g))

        prev_fitnesses = np.copy(fitnesses)
        prev_population = np.copy(population)

    logger.log("Best Ind Weights <")
    for v in best_solution: logger.log(str(v))
    logger.log(">")

def reinforce(results, ind, env):
    global get_gradient
    baseline = results[1] #Total rewards mean
    trajs = results[3]  # List of trajectories that make up the batch
    rewards = results[4] #List of the total reward that is assocaited with each trajectory
    sum_gradient = np.zeros((len(ind), 1))

    rets = []

    for index in range(0, trajs.__len__()):
        traj = trajs[index]
        traj_total_reward = rewards[index]

        traj_obs = np.zeros((len(traj), env.observation_space.shape[0]))
        actions = np.zeros((len(traj), env.action_space.shape[0]))

        i = 0
        for traj_ob in np.array(traj)[:, 0]: #Generate list of trajectory observations
            actions[i] += np.array(traj)[:, 1][i]
            traj_obs[i] += traj_ob
            i += 1

        ret = traj_total_reward - baseline
        rets.append(ret)

        delta = np.asarray(get_gradient(traj_obs, actions, [ret]))
        sum_gradient = np.add(sum_gradient, delta.T)

    mean_gradient = sum_gradient / len(trajs)

    maxVal = 0
    for val in mean_gradient:
        if abs(val) > maxVal:
            maxVal = abs(val)

    mean_gradient = np.divide(mean_gradient, maxVal)

    new_ind = np.asarray(ind).reshape(mean_gradient.shape)
    new_ind = np.add(new_ind, (ALPHA * mean_gradient))
    solution = new_ind.reshape(len(ind), ).tolist()

    if np.isnan(solution).any():
        print("Returned Parent")
        return ind

    return solution

def es_eval(env, env_id, pi,
            solution, best_solution_so_far, iterations,
            stochastic,
            horizon, seed, best_fitness, id, set_from_flat):
    """
    Simulate the env and policy for max n iterations
    """
    set_from_flat(solution)
    rewards = []
    trajs = []

    global timesteps_so_far
    global STEPS_SO_FAR
    global MAX_STEPS

    for _ in range(iterations):
        traject = []
        total_reward = 0
        total_step = 0
        ob = env.reset()

        while True:
            ac = pi.act(stochastic, ob)

            min = -1
            max = 1

            if action_clip == True:
                for i in range(0, len(ac)):
                    if ac[i] < min:
                        ac[i] = min
                    if ac[i] > max:
                        ac[i] = max

            next_ob, rew, done, _ = env.step(ac)

            total_reward += rew
            timesteps_so_far += 1
            total_step += 1

            traject.append([ob, ac, next_ob, rew])
            ob = next_ob

            global MUT_POWER
            global INIT_MUTATE
            global RAND_MUT_POWER
            RAND_MUT_POWER = INIT_MUTATE - INIT_MUTATE * (STEPS_SO_FAR / MAX_STEPS)

            if RAND_MUT_POWER < 0.001:
                RAND_MUT_POWER = 0.001  # Lower bounds

            #Alpha decay:
            STEPS_SO_FAR +=1
            global ALPHA
            global INIT_ALPHA
            ALPHA = INIT_ALPHA-INIT_ALPHA*(STEPS_SO_FAR/MAX_STEPS)

            if ALPHA < 0.001:
                ALPHA = 0.001 #Lower bounds

            if total_step > 3000:
                break

            if done:
                rewards.append(total_reward)
                trajs.append(traject)
                break

    return id, np.mean(rewards), np.sum(rewards), trajs, rewards

