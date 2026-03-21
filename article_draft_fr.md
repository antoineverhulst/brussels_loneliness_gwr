# Article

# Introduction

En 1995, le sociologue américain Robert Putnam alarmait sur le déclin des liens sociaux et de l’engagement civique aux Etats-Unis. Dans son essai “***Bowling Alone: America's Declining Social Capital*”** et le livre qui suivit 5 ans plus tard (**”*Bowling Alone: The Collapse and Revival of American Community”),*** l’auteur revient sur ******l’importance du tissu social pour la résilience et l’efficacité d’une société pour différentes variables sociales et économiques, du taux de pauvreté à la consommation de drogue en passant par le niveau de santé.  

La même année, une vague de chaleur à des niveaux encore jamais vu frappa la métropole de Chicago et entraina un nombre de morts inattendu et déstabilisant. Parmi ceux-ci, le part anormalement élevés de personnes âgés vivant seuls interpella le sociologue **Eric Klinenberg, qui publia en 2022 une analyse sociologique de l’événement (*Canicule. Chicago, été 1995. Autopsie sociale d’une catastrophe)*** mettant en évidence le lien entre mortalité et isolement social. 
L’ouvrage de Klinenberg souligne la fragilité de certaines populations face à des catastrophes climatiques et alerte sur la résistance de nos sociétés face à des événements extrêmes de plus en plus fréquent. Cependant, la solidité d’une population ne peut-être mesurer qu’à des moments extraordinaires, et, comme le souligne Robert Putnam, le délitement de la cohésion sociale détériore de nombreux indicateurs permanents du bien-être général.  

Si ces deux analyses datent et concernent les Etats-Unis, il est très probable que les mêmes conclusion peuvent être faites sur l’Europe de 2026, du fait des différences crises économiques sociales qu’a connu le continent, mais également de la propagation du numérique. Une récente étude la Commission Européenne sur le sujet documente ces phénomènes à travers une enquête menée dans les 27 pays de l’Union Européenne (Schnepf, D’Hombres & Mauri, 2024). Il en ressort qu’un **peu plus d’un européen sur 10 se sent seul** et varie d’un pays à l’autre allant de moins de 10% en Espagne à 20% en Grèce. 

Cependant, plus qu’un caractéristique culturelle à l’échelle d’une nation, l’isolement social dépend surtout l’environnement local immédiat. Comme le montra très bien Eric Klinenberg à Chicago, 2 quartiers pourtant très proche l’un de l’autre peuvent avoir un tissu social très différent et donc résister plus ou moins bien à des évènements dramatiques. 

Quels caractéristiques locales sont le plus liées à l’isolement sociales? Lorsqu’il n’est pas possible de passer pour des enquêtes pour étudier cette question de nombreuses études utilisent le taux de personnes vivant seul (Fritsch, Riederer & Seewann, 2023), Burlina & Rodríguez-Pose, 2023). Si vivre seul ne veut pas nécessairement dire être isolé, il en constitue néanmoins d’un facteur de risque qu’il s’agit de prendre en compte (Lykes & Kemmelmeier, 2013). 10 ans avant de couvrir la canicule à Chicago, Eric Klinenberg avait déjà souligné l’importance de l’environnement urbain aux Etats-Unis pour s’épanouir dans une vie seule et ne pas se retrouver isoler (Klinenberg 2015). Il s’agit donc d’étudier l’évolution des personnes vivant seules en liant avec d’autres facteurs aggravants dans leurs environnements immédiats pour comprendre une partie de l’isolement social.

C’est ce qu’on entrepris récemment Benassi et Iglesias-Pascual (2015), qui ont analysé la distribution spatiales des personnes vivant seules dans les 4 plus grandes villes espagnoles. Dans leur article, les deux auteurs analysent le lien entre vivre seul et 4 autres variables: le revenu moyen par ménage, le pourcentage d’immigrant d’un pays européen, le pourcentage de la population entre 25 et 44 ans, le pourcentage de la population âgé de 65 ou plus. Grâces aux 2 méthodes utilisées, la régression linéaire (OLS) et la Geographical Weight Regression (GWR). leur article montre que ces liens varient d’une ville à l’autre et au sein des villes elle-même. Les deux auteurs identifient des poches de villes où l’association entre ces variables sont particulièrement élevés, soulignant une certaine forme de ségrégation spatiale où les personnes seules ont tendance à être plus pauvres, ou plus vieilles dans certaines partie de la ville seulement. 

Ce type d’étude met en lumière le distribution de l’isolement sociale à l’intérieur d’une ville et y associe d’autres phénomènes, permettant de cibler et affiner les politiques urbaines. Il peut dont s’agir d’un outil de diagnostic particulièrement intéressant pour aider les villes à contrer l’isolement social et ses effets, tels que documentés par la Commission Européenne et Eric Klinenberg. Pour évaluer cette hypothèse, le reste de cet article adaptera la méthodologie de Benassi et Iglesias-Pascual (2015) à la la ville de Bruxelles, utilisant les données mis en disposition en open data par le service d’urbanisme de la ville sur monitoringdesquartiers.brussels.

# Source de données

Le service de planification de la Région Bruxelles Capitale, perspective.brussels, met à disposition sur son site [monitoringdesquartiers.brussels](http://monitoringdesquartiers.brussels) plus de cents indicateurs pour différents années sur les 3 dernières décennies pour 3 niveaux géogaphique (commune, quartiers, secteurs statistiques, voir le site de [StatBel](https://statbel.fgov.be/fr/propos-de-statbel/methodologie/classifications/secteurs-statistiques) pour plus d’information). Pour la présente analysé, les indicateurs suivant ont été utilisé aux niveau du secteur statistique:

- Part des isolées de 30 ans et plus dans le totals des ménages privées (utilisé comme proxy pour représenter l’isolement social) en 2021
- Distance moyenne d’accès aux 4 biens de base (pain, viande, alimentation générale, pharmacie) en 2021
- Part de la population originaire de l’Europe des 14 (hors Belgique) en 2021
- Part des 30-44 ans dans la population en 2021
- Part des plus de 65 ans et plus dans le total des ménages privées en 2021
- Part des logements ayant accès à un jardin privés en 2021
- Revenu equivalent médian après impôts en 2021
- Taux annuel moyen de croissance de la population entre 2016 et 2021

Ces variables sont décrites en détail sur le site du monitoring des quartiers. La table 1 résume certaines statistiques pour chacune de ces variables et la figure 1 présente la corrélation entre chacune d’entre elle. 

[summary_statistics_table.png ]

*Tableau 1: Statistique des variables utilisées*

Comme nous pouvons le voir dans la matrice de corrélation, certains variables sont déjà très fortement corrélées, notamment le taux de végétalisation. Bien qu’ayant une très forte corrélation avec le revenu, la distance moyenne aux 4 biens de bases et la part des logements ayant accès à un jardin privés, nous décidons de gardés cet indicateur car il représente un prisme différents sur la qualité de vie des personnes vivant sur le territoire.

[correlation_matrix.png]

*Figure 1: Corrélations des variables utilisées*

# L’isolement à Bruxelles

Comme illustré par la figure 2, l’isolement des plus de 30 ans est particulièrement élévé dans le sud du Pentagone, notamment dans le quartier des Marolles qui apparait en rouge foncé,  et les communes de Saint-Gilles, Ixelles et Etterbeek. Ces quartiers sont connus pour acceuillir des quartier dynamiques et une population jeune et aisées, en particulier autour des instituions européennes. 

[part-des-isoles-de-30-ans-et-plus-dans-le-total-des-menages-prives.png]

*Figure 2: Part des isolés de moins de 30 ans dans le total des ménages privés*

[https://monitoringdesquartiers.brussels/Indicator/IndicatorPage/2316?tab=sheet-link#sheet](https://monitoringdesquartiers.brussels/Indicator/IndicatorPage/2316?tab=sheet-link#sheet)

L’isolement à Bruxelles semble donc surtout liée par une population aisées et dynamique, associés aux quartiers tendances et attractifs.

Pour documenter plus en détails cette hypothèse, les mêmes méthodes que Benassi et Iglesias-Pascual (2015) peuvent être utilisées pour analyser la relation entre l’isolement, telle que mesurée par la *Part des isolés de moins de 30 ans dans le total des ménages privés* et les autres variables retenues.

# OLS

La régression linéaire est une méthode statistique classique, permettant de trouver les coefficients des variables qui expliquerait au mieux les résultats. Autrement dit, cette méthode trouvent des coefficients *communs à toutes les observations,* tels que la somme des différences entre la variable prédite et la variable réelle soit la plus basse possible. Pour faciliter la comparaison entre variable, chaque variable a été standardisés, c’est à dire qu’elles présente une distribution similaire autour d’une moyenne de 0 et d’un écart type 1.

Les coefficients obtenues sont affichés table 3. Pour chacun d’entre eux, il est indiqué sa p-value, c’est à dire le % de chance que ces coefficients ne soit pas dû au hasard. De manière générale, on dit qu’un coefficient est significatif seulement si il y a une p-value de 5% ou moins. Le modèle OLS attribue un coefficient significatifs pour 9 des 13 variables, ce qui est plutôt et explique un R2 de 73%, qui représente la part de la variabilité de la variable dépendante expliqué par notre modèle. 

[Loneliness in Brussels/outputs/figures/ols_results_table.png]

*Tableau 2: Résultat de la régression linéaire*

Cependant, avant d’aller plus loin, il est important de noter que ce modèle ne respecte pas certaines hypothèses qui rend le calcul valide, à savoir la normalité des résidus (la différence entre la variable dépendante et la variable prédite) et la non-multicolinéarité (le fait que certains variables peuvent être expliqué par d’autres variables). 

De plus, un test de Moran’s I sur la variable dépendante et les résidus révèle la présence d’autocorrelation spatiale. Concrètement, cela veut dire que si un secteur statistique présente une forte part de personnes de plus de 30 ans isolées, les secteurs statistique voisins auront une forte de chance de l’être aussi. De même, si le modèle OLS produit des bonnes ou mauvaise prédictions éloignés dans un secteur statistiques, il aura tendance à être aussi bon ou mauvais dans les secteurs environnants.

# GWR

Une autre méthode doit donc être utilisée pour prendre en compte la dépendance entre un secteur et ses voisins et ainsi prédire le lien spécifique entre le taux de personnes isolées et les autres variables pour chaque secteur, indépendemment de leurs voisins. C’est précisemment ce que fait le modèle Geographical Weight Regression (GWR).

Pour chaque secteur de Bruxelles, la GWR crée un coefficient par variable. Ce coefficient représente l’association entre le taux d’isolement et la variable par secteur. Par exemple, si un quartier a un coefficient élevé positif pour la variable revenu, cela veut dire que l’isolement dans ce quartier, par rapport à d’autres quartiers, peut être expliqué par la présence de haut revenus. Ces indicateurs nous renseignent sur la qualification de l’isolement social et la façon dont il se répartit dans l’espace bruxellois. 

Parmi les 6 variables, toutes sont significativement associés à la part d’individus isolées (Voir Tableau 3) . La variables avec la plus fort associations est la part de personnes de plus de 65 ans, qui est positive partout, ce qui s’explique par le lien entre vivre seule et personnes âgées. La part de logement avec un jardin privée est également fortement lié mais négativement en particulier dans les quartiers denses du centre, où le peu de maison avec jardin privés doivent être surtout le logement de ménages.

[gwr_synthesis_table.png]

*Tableau 3: Résultat de la GWR*

[gwr_local_r2.png]

*Figure 3: R2 expliqué par le modèle GWR pour chaque secteur statistique.* 

Le modèle prédit assez bien la variable dépendante à l’ouest de la ville, à part le quartier d’Alma tandis qu’il performe moins bien à l’est, en particulier à Anderlecht (voir image 2). 

Comme le montre la carte des coefficients de l’image 3, l’effect des coefficients varie d’un quartier à l’autre, et montre des regroupements urbains intéressants. Tout d’abord, l’influence de la part de personnes issues de l’Europe des 14 est négative à la périphérie est et ouest de la ville, tandis qu’elle est positive dans le centre, sud et nord. Dans le centre, les quartiers qui attirent les personnes expatriées doivent être les mêmes que celles qui attirent les personnes vivant seules tandis que dans les quartiers périphériques, les personnes expatriées doivent avoir tendance à s’installer en famille dans des quartiers résidentiels, et ainsi éviter les endroits où préféreraient s’installer les personnes vivant seules. On remarque une exceptionnelle association dans les quartier nords de la ville, problablement du fait d’expatriés travaillant pour l’Otan.  

De façon similaire, on remarque une forte sensibilité de la part de personnes vivant seules à la distance aux 4 biens de bases dans le centre et nord de la ville, là où il est négatif ou proche du nulle dans le reste de la ville. Ces quartiers sont principalement des quartiers de bureaux ou de passages avec très peu d’accès au bien de base. On peut supposer que les personnes s’installant dans ces quartiers ne peuvent qu’être que des personnes vivant seules et potentiellement intéresser par le travail ou d’autres activités solitaire. Pour le dire autrement, si ces quartiers avaient plus biens de bases, ils pourraient également attirer des couples et des familles, et le coefficient serait moins élevés. 

Les autres coefficients sont ensuite plus homogènes et confirme une association forte et sur tout le territoire entre la part de personnes vivant seules et la part de personnes de plus de 65 ans et et une faible part de jardin privé. Dans une moindre mesure, la part de personnes vivant seule dépend également de la part des personnes entre 30 et 44 ans et le faible niveau de revenu mais avec quelques exception, comme à Berchem Sainte-Agathe. Enfin le lien avec croissance de la population est relativement faible mais existant et varie seulement les quartiers, mettant en évidence là où la croissance de la population est le fait des ménages ou d’individus. 

[gwr_local_coefficients.png]

*Figure 4: Coefficients GWR par secteurs statistiques*

Notons qu’à travers ces images, 2 histoires d’isolement se dégagent. 

Tout d’abord,  comme suggéré la figure 1, l’isolement des quartiers à l’est et au sud du Pentagone est plus sensible que dans le reste de la capitale. Dans ces secteurs, l’isolement est fortement associé aux population des plus 65 ans et des 30-44 ans (figure 3.c), mais aussi des plus 65 ans (figure 3.d) et assez peu à la présence de jardin privé et d’accès aux 4 biens de bases, tout en restant liés à des populations à plus faibles revenues (figure 3.f ). Dans ces quartiers, l’isolement est problablement le fait de personnes dynamiques socialement prête à sacrifier une partie de leur revenu pour habiter des quartiers centraux et riches en anémité. S’il faut toujours rester vigilant, des mesures préventives et d’informations lors d’événements climatiques devraient dans ces quartiers bien connectés. 

Cependant, un autre quartier transparait dans les cartes de la figure 2 , avec des tendances fort différentes. La part des personnes vivant seules autour du quartier nord et le canal est fortement liés à la présence des plus de 65 ans et la distance aux 4 biens de bases. L’isolement y est également fortement lié au fait de ne pas avoir accès à un jardins privés. Enfin, l’association est négative avec la croissance de la population. Dans ces quartiers plus précarisés que le sud de Bruxelles, l’isolement est donc plus le fait de personnes âgées et et loin des 4 biens de bases, vivant dans des habitations sans jardins susceptible de provoquer une vulnérabilité, notamment en tant de forte chaleur comme à Chicago. Ces quartiers sont donc prioritaires pour non seulemnt mettre en places des mesures préventives et d’informations, mais également des mesures actives lors d’événements dangereux.

# Conclusion

L’isolement est important à mesurer car il est lié à de nombreuses vulnérabilités économiques, sociales et sanitaires. En particulier, les crises climatiques tels que les canicules et les innondations présentent moins de risques dans les quartiers avec un tissu social robuste (Klinenberg, 2024). 

L’analyse spatiale de l’isolement est parfois difficile à analyser avec les méthodes statistiques traditionnelless tels que la régression linéaire qui ne prend pas en compte les dépendances entre secteur géographique. 

A l’image Benassi et Igliesias (2025) pour les métropoles espagnoles, la Geographical Weight Regression (GWR) peut être utilisée pour mieux caractériser l’isolement dans la Ville de Bruxelles. A partir des données de perspective.brussels, cette méthode statistique a permis d’analyser l’influence de 6 variables sur le taux de personnes de plus de 30 ans vivant seul, quartier par quartier. 

Si la GWR a confirmé que l’association entre isolement et personnes âgées et accès à un jardin privées est présente sur tout le territoire bruxellois, elle a mis en évidence des différences entre les quartiers. 

Tout d’abord, la part de personnes isolées est particulièrement liés à la distance aux 4 biens de bases dans les quartiers d’affaires du centre et du nord, tandis que la présence d’expatriés de l’Europe des 14 est positivement lié au centre de Bruxelles mais négativement à l’est et à l’ouest, montrant différente dynamique d’insertion de ces populations dans la capitale. 

Enfin, la GWR décrit plus en détails 2 dynamiques d’isolement à Bruxelles. L’une, dans le sud de Bruxelles, est fortement lié à des populations âgés et active (30-44ans), mais peu à des questions d’accès et est probablement lié à une volonté de vivre dans ces quartiers bien connectées. Pour ces personnes vivant seuls, le niveau de vulnérabilité est donc plus faible et, lors d’événements extrêmes, des politiques de préventions et d’information devraient suffire limiter les dégâts. 

Cependant, un autre cas d’isolement apparaît au nord de Bruxelles et le long du canal. Celui ci reste lié à la présence de personnees âgés, mais plus autant à la population active, et est en revanche fortement associé à un manque d’accès à des jardins privés et aux 4 biens de bases. Ces quartiers représente population particullièrement vulnérable aux événements extrêmes, notamment les canicules et une politique plus actives, par exemple de passage, de distribution d’eaux ou d’accès à des espaces réfrigéés (Klinenberg, 2) serait recommandé.  

# Ressources

- Audard, Frédéric and Le Campion, Grégoire and Pierson, Julie (2024). La régression géographiquement pondérée : GWR. *Rzine*. https://doi.org/10.48645/wk1m-hg05.
- Benassi, F., & Iglesias-Pascual, R. (2025). A local regression approach to studying single-person households and social isolation in the main Spanish cities: a new pathway of socio-spatial polarization? Annals of Operations Research. [https://doi.org/10.1007/s10479-025-06595-8](https://doi.org/10.1007/s10479-025-06595-8)
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery Rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B (Statistical Methodology), 57(1), 289–300. [https://doi.org/10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- BULTEAU J, Le Boennec R., Feuillet T, 2018. Spatial Heterogeneity of Sustainable Transportation Offer Values: A Comparative Analysis of Nantes Urban and Periurban/Rural Areas (France). In : *Urban Science* [en ligne]. 2018. Vol. 2, n° 1, pp. 1‑14. Disponible à l'adresse : [https://doi.org/10.3390/urbansci2010014](https://doi.org/10.3390/urbansci2010014).
- Burlina, C., & Rodríguez-Pose, A. (2023). Alone and lonely: The economic cost of solitude for regions in
Europe. Environment and Planning a: Economy and Space, 55(8), 2067–2087. [https://doi.org/10.1177/](https://doi.org/10.1177/)
0308518X231169286
- Dormann, C. F., Elith, J., Bacher, S., Buchmann, C., Carl, G., Carré, G., Marquéz, J. R. G., Gruber, B., Lafourcade, B., Leitão, P. J., Münkemüller, T., McClean, C., Osborne, P. E., Reineking, B., Schröder, B., Skidmore, A. K., Zurell, D., & Lautenbach, S. (2012). Collinearity: a review of methods to deal with it and a simulation study evaluating their performance. Ecography, 36(1), 27–46. [https://doi.org/10.1111/j.1600-0587.2012.07348.x](https://doi.org/10.1111/j.1600-0587.2012.07348.x)
- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2003). Geographically weighted regression: The analysis
of spatially varying relationships. John Wiley & Sons.
- Fritsch, N.-S., Riederer, B., & Seewann, L. (2023). Living alone in the city: Differentials in subjective wellbeing among single households 1995–2018: Applied research in quality of life. Springer; International
Society for Quality-of-Life Studies, 18(4), 2065–2087. [https://doi.org/10.1007/s11482-023-10177-wy](https://doi.org/10.1007/s11482-023-10177-wy)
- Kim, Y.-K., & Kim, D. (2024). Role of social infrastructure in social isolation within urban communities.
Land, 13, 1260. [https://doi.org/10.3390/land13081260](https://doi.org/10.3390/land13081260)
- Klinenberg, E. (2012). Going solo: The extraordinary rise and surprising appeal of living alone. Penguin.
- Klinenberg, E. (2022). Canicule. Chicago, été 1995. Autopsie sociale d'une catastrophe. Lyon, Editions deux-cent-cinq, coll. « A partir de l'Anthropocène », 2022, 415 p., trad. Marc Saint-Upéry, ISBN : 978-2-919380-43-5.
- Lykes, V. A., & Kemmelmeier, M. (2013). What predicts loneliness? Cultural difference between
individualistic and collectivistic societies in Europe. Journal of Cross-Cultural Psychology.
- Schnepf, S. V., D’Hombres, B., & Mauri, C. (2024). Loneliness in Europe. In *Population economics*. [https://doi.org/10.1007/978-3-031-66582-0](https://doi.org/10.1007/978-3-031-66582-0)