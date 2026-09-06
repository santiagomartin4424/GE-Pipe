#! /usr/bin/env python

# PonyGE2
""" Python GE implementation for Pipeline Evolution """

import os
# MANDATORY to Define this before importing numpy, sklearn, scipy, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from utilities.algorithm.general import check_python_version
check_python_version()
from stats.stats import get_stats, stats
from algorithm.parameters import params, set_params
from utilities.fitness.error_metric import balanced_accuracy
from utilities.fitness.get_data import get_data
from utilities.fitness import data_cache
from utilities.stats import trackers
from threadpoolctl import threadpool_limits
from queue import Empty
import numpy as np
import sys, re, time, signal, multiprocessing
# Set spawn execution method before any sub-thread creation
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    multiprocessing.set_start_method('fork', force=True)


# GLOBAL VARIABLES/PARAMETERS
MAX_TIME_FOR_EVOLUTION = 3600 # Assign 1 hour of budget for the evolutionary process. 3600
TIMEOUT_SECONDS_REFIT = 1200 # Assign 20min of budget for the re-fit with the winning pipeline, using all the train partition. 1200

# Global variable to register if the timeout was triggered
EVOLUTION_TIMEOUT_TRIGGERED = False

#------------------------------------------------------------------------------------------------------------------
# 1. Defition of the exception for the controlled interruption
class TimeoutException(Exception):
    pass



def worker_fit(pipeline, X, y, conn):
    """
    The isolated worker, to execute the re-fit. 

    Using subprocess parallelization, not threads. This way if this process
    exceeds the time limit, it can be killed in a clean way from the execute_refit_with_timeout() function.

    It puts the n_jobs=-1 internal parameter for the Scikit-Learn algorithms.
    """
    
    # 1. Assign explicit parallelism in the scikit-learn algorithms that support n_jobs
    try:
        if hasattr(pipeline, "set_params"):
            # If the pipeline or their steps accept n_jobs, it is configured to use all the cores (-1)
            params_to_update = {}
            for step_name, step_obj in pipeline.named_steps.items():
                if hasattr(step_obj, "n_jobs"):
                    params_to_update[f"{step_name}__n_jobs"] = -1
            if params_to_update:
                pipeline.set_params(**params_to_update)

        # 2. Execute the fit
        pipeline.fit(X, y)
        conn.send(("success", pipeline))     #return the dict with success and the pipeline

    except Exception as e:
        conn.send(("error", str(e)))         #return error, and the type of the error

    finally:
        conn.close()



def execute_refit_with_timeout(pipeline, X_train, y_train, timeout_seconds=1200):
    """This function will create an isolated process to execute the re-fit. If it exceeds the time limit, it is killed."""
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(
        target=worker_fit, 
        args=(pipeline, X_train, y_train, child_conn)
    )
    p.start()

    # El proceso hijo debe cerrar su copia del extremo del padre si no lo usa,
    # y el padre debe cerrar su copia del extremo del hijo.
    # Cierre de la copia no utilizada en el padre. El padre no necesita el extremo de envio del hijo (child_conn).
    child_conn.close()
    
    if parent_conn.poll(timeout=timeout_seconds):
        try:
            status, payload = parent_conn.recv()
            parent_conn.close() # Cierre del extremo del padre antes de recolección
            p.join()

            if status == "success":
                print("✅ The re-fit was successful")
                return payload, True
            else:
                print(f"⚠️ Error during re-fit: {payload}")
                return None, False
        except Exception as e:
            print(f"⚠️ Exception reading from pipe: {e}")
            parent_conn.close()
            p.kill()
            p.join()
            return None, False
    else:
        mins = timeout_seconds / 60.0
        print(f"⏱️ TIMEOUT: The re-fit exceeded the limit of {mins:.2f} minutes ({timeout_seconds}s).")
        print("Closing re-fit process and any sub-processes...")

        # 1. Cerrar el tubo en el padre primero para evitar bloqueos de I/O
        parent_conn.close()

        try:
            import psutil
            parent_proc = psutil.Process(p.pid)
            children = parent_proc.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except Exception:
            pass

        # 3. Eliminar el proceso principal del re-fit
        p.kill()
        p.join()

        return None, False





# 3. Handler for the time exception Inside the evolutionary loop (search_loop), and so that the execution flow goes to the except block (in the fit method of the MyEstimator class, defined later).
def sigalrm_handler(signum, frame):
    global EVOLUTION_TIMEOUT_TRIGGERED
    EVOLUTION_TIMEOUT_TRIGGERED = True
    signal.alarm(0)  # Deactivates any alarm
    print("\n⚠️ [DETECTED TIMEOUT] Time limit reached for the evolutionary process. Received SIGALRM signal. Stopping evolution...")
    raise TimeoutException("Time limit reached in the evolutionary process (search loop).")



# 4. Function to clean the Pool of processes once the evolutionary process (search loop) has finished.
def _cleanup_pool():
    """Función auxiliar para cerrar el Pool de la evolución de forma limpia."""

    # 2. Close the Pool of multiprocessing, cleaning their queues safely.
    if params.get('MULTICORE') and 'POOL' in params and params['POOL'] is not None:
        try:
            params['POOL'].terminate() # Tells the Pool to stop accepting tasks.
            # NO se llama a params['POOL'].join() aquí porque si las colas (Queues) 
            # de IPC quedaron llenas tras el SIGALRM, join() se bloquea indefinidamente.
        except Exception as e:
            print(f"Error closing the Pool: {e}")
        finally:
            params['POOL'] = None      # Delete the reference so that it doesn't interfere with the memory of future executions.

    # 3. Cleaning of the residual processes with psutil, without killing the father.
    try:
        import psutil
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)        # We obtain all the children before sending any signal
                
        if children:
            for child in children:
                try:
                    child.terminate()   # First, send SIGTERM to all the children
                except psutil.NoSuchProcess:
                    pass                

            gone, alive = psutil.wait_procs(children, timeout=1.5)   # Wait a maximum time of 1.5 seconds so that they close properly.
            
            # Kill the survivals with SIGKILL (-9)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

    except Exception as e:
        print(f"Warning during psutil cleanup: {e}")
#------------------------------------------------------------------------------------------------------------------






class MyEstimator:

    def __init__(self):
        # Class attributes initialized
        self.individuals = None
        self.best_pipeline = None


    @staticmethod
    def clean_phenotype(phenotype):
        """Cleans the phenotype to ensure unique names for a pipeline, in the steps of sklearn."""
        def rename_steps(match):
            """Searches for repeated step names, and adds a numerical suffix.
            This function is called in re.sub (outside this function, in clean_phenotype()). We search
            the repeated step-names, and modify them (we add a number at the end).
            
            :param match: The regex triggered, with the text that will be modified.
            :return: The regex (string) modified.
            """

            name = match.group(1)    # We advance the string one position. To obtain instead of '(scaler,' just 'scaler,'
            rename_steps.counter[name] = rename_steps.counter.get(name, 0) + 1    # we store how many times this name has appeared before.
            return f"('{name}_{rename_steps.counter[name]}',"  # we return the modified string
        
        rename_steps.counter = {}
        fixed_phen = re.sub(r"\((imputer|scaler|augmentation|selection|extraction|balance|clf),", rename_steps, phenotype)  #we obtain the fixed phenotype (because we cannot have several steps named the same way, in a python Pipeline)
        return fixed_phen


    def fit(self, X_train, y_train):
        global EVOLUTION_TIMEOUT_TRIGGERED
        EVOLUTION_TIMEOUT_TRIGGERED = False

        # 1. Define the signal handler for SIGALRM (internal clock) and SIGTERM (extern signal, from awc_script.sh)
        signal.signal(signal.SIGALRM, sigalrm_handler)
        
        # 2. Stablish the alarm. Assign time budget for the evolutionary process (search_loop)
        signal.alarm(MAX_TIME_FOR_EVOLUTION)

        try:
            # 1. Start normal evolution.
            self.individuals = params['SEARCH_LOOP']()  # Returns the last generation if it ends on time
            get_stats(self.individuals, end=True)
            if not EVOLUTION_TIMEOUT_TRIGGERED:
                print("✅ Evolution completed successfully inside the time limit.")  
            _cleanup_pool()


        except TimeoutException:
            print("⏱️ Evolution TIME ended. Cleaning resources and Retrieving the best historical individual...")
            
            # --- ROBUST CLEANING OF SUBPROCESSES ---
            # If the MULTIPROCESSING flag is set to True in the parameters file, it means that internally PonyGE2 will try to
            # define a Pool to create several subprocesses for each CORE (and parallelize, using the 'multiprocessing' python library).
            # This is good, however when the evolutionary process is ended, we need to kill all those Pool subprocesses.
            _cleanup_pool()
               
        finally:
            # Deactivate alarms and restore the default behaviour of the signals.
            signal.alarm(0) 
            signal.signal(signal.SIGALRM, signal.SIG_DFL)


        
        # ------------------------------------------------------------------------
        # EXTRACTION OF THE BEST INDIVIDUAL (Works both with the Timeout and without it)
        # ------------------------------------------------------------------------
        best_ind = None

        # Option 1: Search the best historical individual registered in the PonyGE2 tracker (it is updated at the end of each
        # generation, in the search_loop.py, because step() at the end calls get_stats(), which also calls get_stats_soo)
        if hasattr(trackers, 'best_ever') and trackers.best_ever is not None:
            if getattr(trackers.best_ever, 'fitness', None) is not None:
                best_ind = trackers.best_ever
                print(f"🎯 Retrieved best individual from trackers.best_ever (Fitness: {best_ind.fitness})")

        # Option 2: If for whatever reason trackers.best_ever = None, search in the population returned by the SEARCH_LOOP
        if best_ind is None and self.individuals:
            evaluated_inds = [ind for ind in self.individuals if getattr(ind, 'fitness', None) is not None]
            if evaluated_inds:
                best_ind = max(evaluated_inds, key=lambda ind: ind.fitness)
                print(f"🎯 Retrieved best individual from last generation (Fitness: {best_ind.fitness})")

        # GUARDS: If after checking both places there is still no valid individual.
        if best_ind is None or not hasattr(best_ind, 'phenotype'):
            print("❌ Error: No valid evaluated individual was found for the re-fit.")
            self.best_pipeline = None
            return False


        # 4. Materialize the winner pipeline, to do fit with the data.
        from fitness.supervised_learning.supervised_learning_pipclean import supervised_learning_pipclean
        fit_obj = supervised_learning_pipclean()    # we execute the same steps as we did in the fitness function file.
        local_env = fit_obj.context.copy()

        fixed_phenotype = self.clean_phenotype(best_ind.phenotype)
        
        try:
            exec(fixed_phenotype, local_env)
            self.best_pipeline = local_env['my_pipeline']   # Here we already have it, as a Pipeline object. Ready for the fit.
        except Exception as e:
            print(f"Error materializing the pipeling: {e}\nPhenotype: {fixed_phenotype}")
            raise
        
        # 5. Final Re-fit with ALL the train dataset (the _train.csv file, not the test.csv. This last is only used at the end, in __main__, to predict)
        # (Time limit for the re-fit defined in the beginning of the file: TIMEOUT_SECONDS_REFIT )
        # We load the SAME file PonyGE2 used for the evolution, in search_loop, for evaluating each individual with the fitness function.
        # This time with 100% of the data PonyGE2 used. During the evolution, the model could only see parts of the training set ("folds").
        y_train = np.nan_to_num(y_train, nan=-1).astype(int)
        print(f"Training final model with: {params['DATASET_TRAIN']}")    
        trained_pipeline, successful = execute_refit_with_timeout(
                    self.best_pipeline, X_train, y_train, TIMEOUT_SECONDS_REFIT
                )

        if successful:
            self.best_pipeline = trained_pipeline
            print("✅ Re-fit completed successfully.")
            return True
        else:
            print("⚠️ The re-fit could not be completed. The test evaluation is cancelled, because the pipeline couldn't be trained.")
            self.best_pipeline = None
            return False

        """
        This is a closed model. Because now, from this moment, the 'best_pipeline' object is no longer a phenotype; it is a python object with coefficients, weights and internal structures calculated.
        It is ready to receive a new piece of data (test partition) and provide a prediction.
        """



    def predict(self, X):
        if self.best_pipeline is None:
            raise Exception("Debes ejecutar fit() antes de predict()")
        return self.best_pipeline.predict(X)








if __name__ == '__main__':
    # 1. Initialize parameters, to be able to read for example params['DATASET_TRAIN']
    set_params(sys.argv[1:])
    
    #---------------------------------------- 2. START TIMER
    start_load = time.perf_counter()
    print(f"\n[TIMER] Starting to load data: {params['DATASET_TRAIN']}...")
    #----------------------------------------

    # Initial load of data (train partition).
    X_train, y_train, X_test, y_test = get_data(params['DATASET_TRAIN'], None)

    #---------------------------------------- END TIMER
    end_load = time.perf_counter()
    total_time = end_load - start_load
    print(f"✅ [TIMER] Load completed in: {total_time:.4f} seconds.")
    print("-" * 40)
    #---------------------------------------- 

    # 3. Execute search_loop, and do REFIT with the best pipeline  -----------------------------------------------------------------------------------------------
    estimator = MyEstimator()
    refit_successful = estimator.fit(X_train, y_train)

    # If the re-fit was cancelled by timeout or error, the execution is finished successfully (code 0)
    # also to allow the Bash script to continue with the next dataset (if there was any).
    if not refit_successful:
        print(" Skipping test metrics saving for this dataset, due to error/timeout in re-fit.")
        sys.exit(0)


    # Successful Pipeline obtained and trained from the population.
    # 2. Locate the TEST file dynamically ------------------------------------------------------------------------------------------------------------------------
    # For example, if DATASET_TRAIN is ".../abalone_train.csv", we search ".../abalone_test.csv"
    train_path = params['DATASET_TRAIN']
    test_path = train_path.replace("_train.csv", "_test.csv")
    
    if not os.path.exists(test_path):
        # Fallback, just in case is .txt or similar. Optional.
        test_path = train_path.replace("train", "test")

    print(f"Loading test data from: {test_path}")

    
    # 3. Load test data and Predict
    X_test, y_test, _, _ = get_data(test_path, None)
    
    # Quick preprocessing of the test labels (the same was done in the loading of the data in the fitness function file)
    y_test_final = np.nan_to_num(y_test, nan=-1).astype(int)

    try:
        y_pred = estimator.predict(X_test)
        y_pred_final = y_pred.astype(int)
    except Exception as e:
        print(f"Error trying to make prediction with the Test partition. {e}\nPhenotype: {estimator.best_pipeline}")
        raise

    # 4. Final Metric (balanced_accuracy) ----------------------------------------------------------------------------------------
    final_score = params['ERROR_METRIC'](y_test_final, y_pred_final)
    
    print("\n" + "="*40)
    print(f"DATASET: {os.path.basename(test_path)}")
    print(f"BALANCED ACCURACY FINAL (TEST): {final_score:.4f}")
    print("="*40 + "\n")


    # 5. Save the result in a file. We read the environment variable sent by Bash. If it does not exists, classic fallback.
    results_dir = os.environ.get("DYNAMIC_RESULTS_DIR", "../results_scores")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    # Name of the file based on the dataset and the grammar.
    g_name = os.path.basename(params['GRAMMAR_FILE']).replace(".bnf", "")
    d_name = os.path.basename(params['DATASET_TRAIN']).replace("_train.csv", "")
    
    res_file = os.path.join(results_dir, f"scores_{d_name}_{g_name}.txt")
    
    # We write the score ('append' mode, to accumulate the executions of this set)
    with open(res_file, "a") as f:
        f.write(f"{final_score:.4f}\n")

    # Explicit clean exit, so that Bash knows the process ended properly.
    sys.exit(0)

