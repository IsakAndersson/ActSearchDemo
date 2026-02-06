from ranx import Qrels, Run

qrels_dict = { "q_1": { "d_12": 5, "d_25": 3 },
               "q_2": { "d_11": 6, "d_22": 1 } }

run_dict = { "q_1": { "d_12": 0.9, "d_23": 0.8, "d_25": 0.7,
                      "d_36": 0.6, "d_32": 0.5, "d_35": 0.4  },
             "q_2": { "d_12": 0.9, "d_11": 0.8, "d_25": 0.7,
                      "d_36": 0.6, "d_22": 0.5, "d_35": 0.4  } }

qrels = Qrels(qrels_dict)
run = Run(run_dict)