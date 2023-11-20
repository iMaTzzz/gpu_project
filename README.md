# Notre encodeur JPEG à nous

Bienvenue sur la page d'accueil de _votre_ projet JPEG, un grand espace de liberté, sous le regard bienveillant de vos enseignants préférés.
Le sujet sera disponible dès le lundi 2 mai à l'adresse suivante : [https://formationc.pages.ensimag.fr/projet/jpeg/jpeg/](https://formationc.pages.ensimag.fr/projet/jpeg/jpeg/).

Vous pouvez reprendre cette page d'accueil comme bon vous semble, mais elle devra au moins comporter les infos suivantes **avant la fin de la première semaine (vendredi 6 mai)** :

1. des informations sur le découpage des fonctionnalités du projet en modules, en spécifiant les données en entrée et sortie de chaque étape ;
2. (au moins) un dessin des structures de données de votre projet (format libre, ça peut être une photo d'un dessin manuscrit par exemple) ;
3. une répartition des tâches au sein de votre équipe de développement, comportant une estimation du temps consacré à chacune d'elle (là encore, format libre, du truc cracra fait à la main, au joli Gantt chart).

Rajouter **régulièrement** des informations sur l'avancement de votre projet est aussi **une très bonne idée** (prendre 10 min tous les trois chaque matin pour résumer ce qui a été fait la veille, établir un plan d'action pour la journée qui commence et reporter tout ça ici, par exemple).

Informations sur le découpage des fonctionnalités du projet en module : 
1. coding => fichier contenant la fonction pour encoder les coefficients DC et AC de chaque bloc dans le bitstream.
            (Entrée : un tableau représentant le bloc à encoder)
2. dct => fichier contenant la fonction pour faire le dct (façon Loeffler) y compris le zigzag.
            (Entrée : une matrice 8x8 (bloc spatiale)
             Sortie : un tableau contenant les valeurs DCT sous la forme zig-zag)
3. decoupe => fichier contenant les fonctions qui découpe l'image en plusieurs MCUs.
            (Entrées : l'image, la largeur et la hauteur de l'image, les tables de Huffman, le bitstream et les valeurs de sous-échantillonnage.)
4. downsampling => fichier contenant la fonction qui effectue le sous-échantillonage 
                   (Entrées : une matrice à sous-échantillonner -> mcu_in
                    Matrice qui représente le paramètre de sortie -> mcu_out
                    valeur horizontale h (soit h2 ou h3) pour Cb ou Cr
                    valeur verticale v (soit v2 ou v3)   pour Cb ou Cr
                    valeur horizontale h1                pour Y
                    valeur verticale v1                  pour Y
                    Sortie : une matrice sous-échantillonnée -> mcu_out)
5. jpeg_header => fichier contenant deux fonctions (écriture header de l'image jpeg grise ou en couleur)
                  (Entrées : nom de fichier ppm, nom de fichier jpeg, largeur et hauteur de l'image ppm, tables de Huffman dc et ac luminance (Y), tables de Huffman dc et ac chrominance (Cb Cr) et les valeurs du sous-échantillonage
                    Return : la structure jpeg)
6. ppm2jpeg => fichier main
7. quantification => fichier contenant la fonction qui quantifie un array après zigzag en divisant chaque terme par la valeur correspondante dans la table de quantification adéquate.
                    (Entrées : array int16_t de taille 64 et un bool qui vérifier si on est dans le cas Y ou CbCr
                    Sortie : array int16_t de taille 64
                    Return : Rien)
8. rgb_to_ycbcr => fichier contenant la fonction qui convertit un pixel RGB à Y ou Cb ou Cr
                  (Entrées : pixel red, green et blue et un enum color_component Y, Cb, Cr 
                    Return : Y ou Cb ou Cr selon le enum)


Répartition des différentes étapes :

1. Récupération des paramètres lus sur la ligne de commande et dans le fichier d'entrée PPM : fait par Matteo ~2h
2. Ecriture des différents marqueurs qui forment l'en-tête JPEG dans le fichier de sortie : fait par Tee Wei ~1h
3. Découpage de l'image en MCUs : en cours par Thibault
4. Encodage de chaque MCU :
    1. Changement de représentation des couleurs : conversion RGB vers YCbCr : fait par Tee Wei ~30min
    2. Compression des composantes Cb et Cr en cas de sous-échantillonnage : fait par tout le monde ~Une journée
    3. Compression de chaque bloc :
        1. Calcul de la transformée en cosinus discrète (DCT) : fait par Tee Wei ~Une journée
        2. Réorganisation zig-zag : fait par Thibault ~1h30 et Matteo ~1h30
        3. Quantification (division par les tables de quantification) : fait par Thibault ~30min
        4. Compression AC/DC et écriture dans le flux : fait par Thibault ~6h et Matteo ~6h

Lundi 2 Mai : Répartition des tâches et création du fichier ppm2jpeg 

Mardi 3 Mai : Création des programmes zigzag, quantify, conversion et DCT.

Mercredi 4 Mai : Les programmes de la veille sont testés et vérifiés. Debut de coding AC/DC.

Jeudi 5 Mai : Toutes les étapes pour la création d'invader ont été effectuées et testées mais quelques bugs sont à corriger puisque l'image ne s'affiche pas comme il faut.

Vendredi 6 Mai : Correction et affichage d'invaders ! Création du découpage en MCU.

Lundi 9 Mai : Le découpage en MCU marche ! Meme avec troncature (Pour l'image grise).

Mardi 10 Mai : Optimisation DCT, Zig-zag. Commencement du découpage en MCU des images en couleurs et du sous-échantillonnage.

Mercredi 11 Mai : On continue l'optimisation (DCT avec l'algorithme de Loeffler) et la fonction downsampling. 

Jeudi 12 Mai : Généralisation sur tous les cas possibles de la fonction downsampling (Fin).
               Petite mise à jour sur l'optimisation de DCT et zigzag.
               Commencement du module jpeg_writer.

Vendredi 13 Mai : Commencement des modules bitstream et huffman.
                  Fin du module jpeg_writer.

Lundi 16 Mai : Fin du module huffman.
               Continuation du module bitstream et de l'optimisation DCT avec l'algorithme loeffler.

Mardi 17 Mai : Toujours sur le bitstream module et de l'optimisation DCT avec l'algorithme loeffler.

Mercredi 18 Mai : Factorisation, amélioration et explication du code.



>[image](dessin_sdd_page-0001.jpg "Structure de données de chaque étape pour invader")

# Liens utiles

- Bien former ses messages de commits : [https://www.conventionalcommits.org/en/v1.0.0/](https://www.conventionalcommits.org/en/v1.0.0/) ;
- Problème relationnel au sein du groupe ? Contactez [Pascal](https://fr.wikipedia.org/wiki/Pascal,_le_grand_fr%C3%A8re) !
- Besoin de prendre l'air ? Le [Mont Rachais](https://fr.wikipedia.org/wiki/Mont_Rachais) est accessible à pieds depuis la salle E301 !
- Un peu juste sur le projet à quelques heures de la deadline ? Le [Montrachet](https://www.vinatis.com/achat-vin-puligny-montrachet) peut faire passer l'envie à vos profs de vous mettre une tôle !