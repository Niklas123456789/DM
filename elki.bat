java -jar elki-bundle-0.7.5.jar KDDCLIApplication ^
-dbc.in sample.csv ^
-algorithm clustering.subspace.PreDeCon ^
-dbscan.epsilon 1.0 ^
-dbscan.minpts 3 ^
-predecon.delta 0.25 ^
-predecon.kappa 100.0 ^
-predecon.lambda 1 ^
-resulthandler AutomaticVisualization