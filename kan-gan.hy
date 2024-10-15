(import torch :as pt)
(import kan)
(require hyrule *)
(import pandas :as pd)
(import matplotlib.pyplot :as plt)

(setv device (pt.device (if (.is-available pt.cuda) "cuda:1" "cpu")))

(defn scale     [xmin xmax x] (/ (- x xmin) (- xmax xmin)))
(defn unscale   [xmin xmax x] (+ (* x (- xmax xmin)) xmin))
(defn trafo     [x msk] (+ (* (pt.log x) msk) (* x (- 1.0 msk))))
(defn untrafo   [x msk] (+ (* (pt.pow 10 x) msk) (* x (- 1.0 msk))))
(defn make-mask [cols msk] (-> (lfor c cols (in c msk)) pt.tensor (.float) (.to device)))

(defn make-dataset [path x-params y-params x-msk y-msk [ratio 0.8]]
  (let [df-raw (pd.read-csv path)
        num    (len df-raw)
        split  (int (* num ratio))
        df     (.sample df-raw num :replace False)
        xs-raw (-> (. (get df x-params) values)
                   pt.from-numpy (.float) (.to device))
        ys-raw (-> (. (get df y-params) values)
                   pt.from-numpy (.float) (.to device))
        xs-trf (trafo xs-raw x-msk)
        ys-trf (trafo ys-raw y-msk)
        xs     (scale (get (pt.min xs-trf :axis 0) 0)
                      (get (pt.max xs-trf :axis 0) 0) xs-trf)
        ys     (scale (get (pt.min ys-trf :axis 0) 0)
                      (get (pt.max ys-trf :axis 0) 0) ys-trf)]
    {"train_input" (.reshape (cut xs None split) -1 (len params-x))
     "train_label"  (.reshape (cut ys None split) -1 (len params-y))
     "test_input"  (.reshape (cut xs split None) -1 (len params-x))
     "test_label"  (.reshape (cut ys split None) -1 (len params-y))})) 

;(setv path     "/home/uhlmanny/Workspace/ganarchist/data/gans.csv"
;      params-x ["v_ds_work" "i_d_max"]
;      params-y ["r_ds_on" "r_g" "g_fs" "v_gs_work" "v_gs_max" "v_th" "c_iss" "c_oss" "c_rss"]
;      dataset  (make-dataset path params-x params-y))
(setv path     "/home/uhlmanny/Workspace/ganarchist/data/volumes.csv"
      params-x ["rth"]
      params-y ["vol"] ;["wb" "lb" "hb" "wfin" "hfin" "nfin" "vol"]
      mask-x   (make-mask ["rth"] params-x)
      mask-y   (make-mask ["vol"] params-y)
      dataset  (make-dataset path params-x params-y mask-x mask-y))

(plt.scatter (-> dataset (get "train_label") (.reshape -1) (.to "cpu") list)
             (-> dataset (get "train_input") (.reshape -1) (.to "cpu") list))
(.grid plt)
(plt.ylabel "Volume")
(plt.xlabel r"$R_{th}$")
(.show plt)

; Hyperparameters
(setv grids       (list (range 3 6)) ; [3 10 20 50 100] ; (list (range 1 15 3))
      width       [(len params-x) (len params-y)]
      steps       100
      k           3
      α           1.0
      λ           1e-15
      λ-entropy   2.0
      noise-scale 0.25
      batch-size  -1
      optim       "LBFGS"
      base-fun    pt.nn.functional.mish
      seed        666)

(defn refine [m gs ls]
  (if (<= (len gs) 0) #(m (tuple (map pt.concat (zip #* ls))))
    (let [mdl (if m (m.refine (get gs 0))
                (kan.KAN :width       width
                         :grid        (get gs 0)
                         :k           k
                         :noise-scale noise-scale
                         :seed        seed
                         :base-fun    base-fun
                         :device      device))
          res (mdl.fit dataset :lamb                 λ
                               :lamb-entropy         λ-entropy
                               :opt                  optim
                               :steps                steps
                               :lr                   α
                               :singularity-avoiding True
                               :batch                batch-size)
          tl  (pt.concat (lfor l (get res "train_loss") (-> l pt.from-numpy (.reshape 1))))
          vl  (pt.concat (lfor l (get res "test_loss")  (-> l pt.from-numpy (.reshape 1)))) ]
      (refine mdl (cut gs 1 None) (+ ls [#(tl vl)])))))

(setv #(mdl #(train-loss valid-loss)) (refine None grids []))

(.plot mdl :in-vars params-x :out-vars params-y) (.show plt)

(plt.plot (list train-loss) :label "train")
(plt.plot (list valid-loss) :label "valid")
(.legend plt)
(.grid plt)
(plt.ylabel "RMSE")
(plt.xlabel "Step")
(plt.yscale "log")
(.show plt)

(defn num-params [ws]
  (if (< (len ws) 2) 0
    (+ (* (get ws 0) (get ws 1)) (num-params (cut ws 1 None)))))

(setv n-params (* (pt.tensor grids) (num-params (list (map (fn [x] (get x 0)) width))))
      train-vs-G (cut train-loss (- steps 1) None steps)
      valid-vs-G (cut valid-loss (- steps 1) None steps))

(plt.plot n-params train-vs-G :marker "o" :label "train")
(plt.plot n-params valid-vs-G :marker "o" :label "valid")
;(plt.plot n-params (* (** n-params -4.0) 100) :ls "--" :color "black" :label r"$N^{-4}$")
(plt.xscale "log")
(plt.yscale "log")
(plt.legend)
(plt.grid)
(plt.xlabel "Number of Prameters")
(plt.ylabel "RMSE")
(plt.show)

(plt.scatter (-> dataset (get "test_label") (.reshape -1) (.to "cpu") list)
             (-> dataset (get "test_input") mdl (.detach) (.clone) (.reshape -1) (.to "cpu") list))
(.grid plt)
(plt.xlabel "Ground Truth")
(plt.ylabel "Prediction")
(.show plt)

(plt.scatter (-> dataset (get "test_input") (.reshape -1) (.to "cpu") list)
             (-> dataset (get "test_label") (.reshape -1) (.to "cpu") list)
             :label "Ground Truth")
(plt.scatter (-> dataset (get "test_input") (.reshape -1) (.to "cpu") list)
             (-> dataset (get "test_input") mdl (.detach) (.clone) (.reshape -1) (.to "cpu") list)
             :label "Prediction")
(.grid plt)
(.legend plt)
(plt.xlabel r"$R_{\mathrm{th}}")
(plt.ylabel "Volume")
(.show plt)
