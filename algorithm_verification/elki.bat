java -jar elki-bundle-0.7.5.jar KDDCLIApplication ^
-dbc.in datasets\multiple-gaussian-2d_unlabeled.csv ^
-algorithm clustering.subspace.PreDeCon ^
-dbscan.epsilon 1 ^
-dbscan.minpts 3 ^
-predecon.delta 0.25 ^
-predecon.kappa 100 ^
-predecon.lambda 1 ^
-resulthandler AutomaticVisualization