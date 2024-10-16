(require hyrule *)
(import torch :as pt)
(import kan)
(import pandas :as pd)
(import matplotlib.pyplot :as plt)
(import functools [partial])

(setv device (pt.device (if (.is-available pt.cuda) "cuda:1" "cpu")))

(defn scale     [xmin xmax x] (/ (- x xmin) (- xmax xmin)))
(defn unscale   [xmin xmax x] (+ (* x (- xmax xmin)) xmin))

(defn trafo     [msk x] (+ (* (pt.log x) msk) (* x (- 1.0 msk))))
(defn untrafo   [msk x] (+ (* (pt.pow 10 x) msk) (* x (- 1.0 msk))))

(defn make-mask [msk cols] (if msk (-> (lfor m msk (in m cols)) pt.tensor (.float))
                                   (-> cols len pt.zeros)))

(defn make-dataset [path params-x params-y mask-x mask-y [ratio 0.75] [seed 666]]
  (let [df-raw (pd.read-csv path)
        num    (len df-raw)
        split  (int (* num ratio))
        df     (.sample df-raw num :replace False :random-state seed)
        msk-x  (-> mask-x (make-mask params-x) (.to device))
        msk-y  (-> mask-y (make-mask params-y) (.to device))
        xs-raw (-> (. (get df params-x) values)
                   pt.from-numpy (.float) (.to device))
        ys-raw (-> (. (get df params-y) values)
                   pt.from-numpy (.float) (.to device))
        xs-trf (trafo msk-x xs-raw)
        ys-trf (trafo msk-y ys-raw)
        xs     (scale (get (pt.min xs-trf :axis 0) 0)
                      (get (pt.max xs-trf :axis 0) 0) xs-trf)
        ys     (scale (get (pt.min ys-trf :axis 0) 0)
                      (get (pt.max ys-trf :axis 0) 0) ys-trf) ]
    {"train_input" (.reshape (cut xs None split) -1 (len params-x))
     "train_label" (.reshape (cut ys None split) -1 (len params-y))
     "test_input"  (.reshape (cut xs split None) -1 (len params-x))
     "test_label"  (.reshape (cut ys split None) -1 (len params-y))})) 

(defn fitter [λ λ-entropy optim steps α batch-size dataset mdl]
  (let [res (mdl.fit dataset :lamb                 λ
                             :lamb-entropy         λ-entropy
                             :opt                  optim
                             :steps                steps
                             :lr                   α
                             :singularity-avoiding True
                             :batch                batch-size)
        tl  (pt.concat (lfor l (get res "train_loss") (-> l pt.from-numpy (.reshape 1))))
        vl  (pt.concat (lfor l (get res "test_loss")  (-> l pt.from-numpy (.reshape 1))))]
    #(tl vl)))

(defn refine [mdl fit grids ls]
  (if (<= (len grids) 0) #(mdl (tuple (map pt.concat (zip #* ls))))
    (let [m (.refine mdl (get grids 0))
          l [(fit m)]]
      (refine m fit (cut grids 1 None) (+ ls l)))))

(defn train [dataset grids width [k 3] [noise-scale 0.25] [base-fun "identity"]
             [λ 0.0] [λ-entropy 2.0] [optim "LBFGS"] [α 1.0] [steps 100]
             [batch-size -1] [seed 666]]
  (let [mdl (kan.KAN :width       width
                     :grid        (get grids 0)
                     :k           k
                     :noise-scale noise-scale
                     :seed        seed
                     :base-fun    base-fun
                     :device      device)
        fit (partial fitter λ λ-entropy optim steps α batch-size dataset)
        los (fit mdl) ]
    (setv mdl.node-scores [])
    (setv mdl.edge-scores [])
    (setv mdl.subnode-scores [])
    (refine mdl fit (cut grids 1 None) [los])))

(defn trace [model n path]
  (let [xs (pt.rand 10 n)]
    (-> model (pt.jit.trace xs) (pt.jit.save path))
    (pt.jit.load path)))

(defn make-predictor [mdl path params-x params-y mask-x mask-y]
  (let [df-raw  (pd.read-csv path)
        xs-raw  (-> (. (get df-raw params-x) values) pt.from-numpy (.float))
        ys-raw  (-> (. (get df-raw params-y) values) pt.from-numpy (.float))
        msk-x   (make-mask mask-x params-x)
        msk-y   (make-mask mask-y params-y)
        x-trafo (partial trafo msk-x)
        y-trafo (partial untrafo msk-y)
        xs-trf  (trafo msk-x xs-raw)
        ys-trf  (trafo msk-y ys-raw)
        x-scale (partial scale (get (pt.min xs-trf :axis 0) 0)
                               (get (pt.max xs-trf :axis 0) 0))
        y-scale (partial unscale (get (pt.min ys-trf :axis 0) 0)
                                 (get (pt.max ys-trf :axis 0) 0))
        _       (-> mdl (.eval ) (.to (pt.device "cpu")))]
    (setv mdl.node-scores      []
          mdl.edge-scores      []
          mdl.subnode-scores   []
          mdl.subnode-actscale (lfor a mdl.subnode-actscale (.to a mdl.device))
          mdl.edge-actscale    (lfor a mdl.edge-actscale    (.to a mdl.device)))
    (fn [x] (with [_ (.no-grad pt)] (-> x x-trafo x-scale mdl y-scale y-trafo)))))

(defn num-params [ws]
  (if (< (len ws) 2) 0
    (+ (* (get ws 0) (get ws 1)) (num-params (cut ws 1 None)))))

(defn plot-model [m px py] (.plot m :in-vars px :out-vars py) (.show plt))

(defn plot-loss [tl vl]
  (plt.plot (list tl) :label "train")
  (plt.plot (list vl) :label "valid")
  (.legend plt) (.grid plt) (plt.yscale "log")
  (plt.ylabel "RMSE") (plt.xlabel "Step")
  (.show plt))

(defn plot-refinement [tl vl s g w]
  (let [n-params (* (pt.tensor g) (num-params (list (map (fn [x] (get x 0)) w))))
        train-vs-G (cut tl (- s 1) None s)
        valid-vs-G (cut vl (- s 1) None s)]
    (plt.plot n-params train-vs-G :marker "o" :label "train")
    (plt.plot n-params valid-vs-G :marker "o" :label "valid")
    ;(plt.plot n-params (* (** n-params -4.0) 100) :ls "--" :color "black" :label r"$N^{-4}$")
    (plt.xscale "log") (plt.yscale "log") (plt.legend) (plt.grid)
    (plt.xlabel "Number of Prameters") (plt.ylabel "RMSE")
    (plt.show)))
