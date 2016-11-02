import single_layer
#clf_q = single_layer.Classifer()
#clf_q.SetPar(2,2)
#clf_q.LearningQ(1,0,300,0.15)

clf_ce = single_layer.Classifer()
clf_ce.SetPar(2,2)
clf_ce.LearningCE(1,0,300,0.15)