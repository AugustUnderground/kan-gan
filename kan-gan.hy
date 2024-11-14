(require hyrule *)
(import torch :as pt)
(import pandas :as pd)
(import matplotlib.pyplot :as plt)
(import util *)

;(setv path     "/home/uhlmanny/Workspace/ganarchist/data/volumes.csv"
;      seed     666
;      params-x ["rth"]
;      params-y ["wb" "lb" "hb" "wfin" "hfin" "nfin" "vol"]
;      mask-x   ["rth"]
;      mask-y   ["vol"]
;      dataset  (make-dataset path params-x params-y mask-x mask-y :seed seed))

(setv path     "./EPC2045.csv"
      seed     666
      params-x ["vg" "vd" "temp"]
      params-y ["M0:2"]
      mask-x   []
      mask-y   ["M0:2"]
      dataset  (make-dataset path params-x params-y mask-x mask-y :seed seed))

 ; [2 4 6 8] ; [3 10 20 50 75] ; (list (range 3 10)) ; (list (range 1 15 3))

(setv grids [1 2 3 4 5]
      width [(len params-x) 2 (len params-y)]
      steps 100
      args { "steps"       steps
             "k"           4
             "α"           1.0
             "λ"           0.002 ; 1e-15
             "λ_entropy"   2.0
             "noise_scale" 0.1 ; 0.25
             "batch_size"  -1
             "optim"       "LBFGS"
             "base_fun"    pt.nn.functional.mish
             "seed"        seed })

(setv #(mdl #(train-loss valid-loss)) (train dataset grids width #** args))

(plot-model mdl params-x params-y)
(plot-loss train-loss valid-loss)
(plot-refinement train-loss valid-loss steps grids width)

(print (equation mdl params-x))

(.cpu mdl)

(with [_ (.no-grad pt)]
  (setv xs (pt.rand 10 (len params-x)))
  (setv _ (mdl xs))
  (setv _(detach mdl))
  (setv _ (mdl xs))
  (setv predictor (make-predictor mdl path params-x params-y mask-x mask-y)))

(with [_ (.no-grad pt)]
  (setv trace (pt.jit.trace predictor xs))
  (pt.jit.save trace "./trace.pt"))

(setv df (pd.read-csv "./Gfs_EPC_EPC2045.txt"))
;(setv df (get df-raw (.all (> (. (get df-raw (+ mask-x ["Id"])) values) 0.0) :axis 1)))
(setv xs (.float (pt.from-numpy (. (get df ["Vgs" "T"]) values))))
(setv ys  (.squeeze (.float (pt.from-numpy (. (get df ["Id"]) values)))))
(setv ys_ (.squeeze (.detach (predictor xs))))

(plt.scatter (.cpu (get xs [(slice None) 0])) (.cpu ys)) (plt.show)
(plt.scatter (.cpu (get xs [(slice None) 0])) (.cpu ys_)) (plt.show)




(setv foo (detach mdl))

(mdl xs)


mdl.__dict__

(setv predictor (make-predictor script path params-x params-y mask-x mask-y))

(setv trace (pt.jit.trace predictor xs))
(pt.jit.save trace "./trace.pt")
(setv script (pt.jit.load "./trace.pt"))
(script xs)

(setv bar (trace foo (len params-x) "./model/trace.pt"))

(setv predictor (make-predictor mdl path params-x params-y mask-x mask-y))


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
