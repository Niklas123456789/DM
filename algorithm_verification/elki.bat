java -jar elki-bundle-0.7.5.jar KDDCLIApplication ^
-dbc.in datasets\multiple-gaussian-2d_unlabeled.csv ^
-algorithm clustering.subspace.PreDeCon ^
-dbscan.epsilon 1 ^
-dbscan.minpts 8 ^
-predecon.delta 0.5 ^
-predecon.kappa 100 ^
-predecon.lambda 2 ^
-resulthandler AutomaticVisualization