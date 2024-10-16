(require hyrule *)
(import torch :as pt)
(import matplotlib.pyplot :as plt)
(import util *)

(setv path     "/home/uhlmanny/Workspace/ganarchist/data/volumes.csv"
      seed     666
      params-x ["rth"]
      params-y ["wb" "lb" "hb" "wfin" "hfin" "nfin" "vol"]
      mask-x   ["rth"]
      mask-y   ["vol"]
      dataset  (make-dataset path params-x params-y mask-x mask-y :seed seed))

(setv grids [2 4 6 8] ;(list (range 3 10)) ; [3 10 20 50 100] ; (list (range 1 15 3))
      width [(len params-x) 1 (len params-y)]
      steps 100
      args { "steps"       steps
             "k"           3
             "α"           1.0
             "λ"           1e-15
             "λ_entropy"   2.0
             "noise_scale" 0.25
             "batch_size"  -1
             "optim"       "LBFGS"
             "base_fun"    pt.nn.functional.mish
             "seed"        seed })

(setv #(mdl #(train-loss valid-loss)) (train dataset grids width #** args))

(plot-model mdl params-x params-y)
(plot-loss train-loss valid-loss)
(plot-refinement train-loss valid-loss steps grids width)

(setv lib ["x" "x^2" "x^3" "x^4" "exp" "log" "sqrt" "tanh" "sin" "tan" "abs"])
(mdl.auto-symbolic :lib lib)
(.symbolic-formula mdl :var ["rth"])
(setv eqn (get (.symbolic-formula mdl) 0 0))
(print eqn)

(setv predictor (make-predictor mdl path params-x params-y mask-x mask-y))

(predictor (pt.rand 10 1))

(setv model (trace predictor (len params-x) "./model/trace.pt"))


(plt.scatter (-> dataset (get "test_label") (get [(slice None None) (slice -1 None)])
                         (.to "cpu") (.reshape -1))
             (-> dataset (get "test_input") (.to "cpu") mdl (.detach) (.clone)
                         (get [(slice None None) (slice -1 None)]) (.reshape -1)))
(.grid plt) (plt.xlabel "Ground Truth") (plt.ylabel "Prediction")
(.show plt)

(plt.scatter (-> dataset (get "test_input") (.reshape -1) (.to "cpu") (.reshape -1))
             (-> dataset (get "test_label") (get [(slice None None) (slice -1 None)])
                         (.to "cpu") (.reshape -1))
             :label "Ground Truth")
(plt.scatter (-> dataset (get "test_input") (.reshape -1) (.to "cpu") (.reshape -1))
             (-> dataset (get "test_input") (.to "cpu") mdl (.detach) (.clone)
                         (get [(slice None None) (slice -1 None)]) (.reshape -1))
             :label "Prediction")
(.grid plt) (.legend plt) (plt.xlabel r"$R_{\mathrm{th}}") (plt.ylabel "Volume")
(.show plt)
