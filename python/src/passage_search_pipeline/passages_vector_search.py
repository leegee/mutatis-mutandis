"""
passage_search_pipeline/passages_vector_search.py
"""

from pathlib import Path
import argparse

import numpy as np
import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.macberth import encode_passage_query
from lib.corpus_logging import get_logger

from passage_search_pipeline import passages_search as ps

logger = get_logger("passages_vector_search")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search historical passages using a MacBERTh query vector."
    )
    parser.add_argument(
        "--query",
         #   default=( "his head and hairs were white as white wool, and as snow, and his eyes were as a flame" ),
         default=( "white hair" ),
         help="Text to encode as the search query.",
    )
    parser.add_argument( "--start-year", type=int, default=1000, help="Earliest publication year (default: 1000).", )
    parser.add_argument( "--end-year", type=int, default=2000, help="Latest publication year (default: 2000).", )
    parser.add_argument( "--representation", default="mean4", help="Passage representation (default: mean4).", )
    parser.add_argument( "--limit", type=int, default=30, help="Maximum number of results (default: 30).", )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")
    if args.limit < 1:
        parser.error("--limit must be >= 1")

    return args


def normalise_vector(vector, description="query vector"):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    if not np.isfinite(vector).all():
        raise ValueError(f"{description} contains NaN or infinity")

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"{description} has zero magnitude")

    return vector / norm


def main():
    args = parse_args()
    lance_dir = Path(LANCE_INDEXES_DIR)

    logger.info("Lance directory: %s", lance_dir)
    logger.info("Query: %s", args.query)
    logger.info(f"Years: {args.start_year}-{args.end_year}")
    logger.info("Representation:%s", args.representation)
    logger.info("Limit: %s", args.limit)

    db = lancedb.connect(str(lance_dir))

    query_vector = normalise_vector(
        encode_passage_query(args.query, args.representation),
        "Text query",
    )

    results = ps.search_passages(
        db=db,
        query_vector=query_vector,
        representation=args.representation,
        start_year=args.start_year,
        end_year=args.end_year,
        limit=args.limit,
        include_vectors=False,
    )

    if not results:
        logger.info("No results found.")
        return

    with get_connection(application_name="revelation-vector-test") as conn:
        for rank, result in enumerate(results, start=1):
            metadata = ps.fetch_passage_text(conn, result)
            logger.info("=============================================================")
            logger.info(
                f"{rank}. year={result['pub_year']} "
                f"distance={result['distance']:.4f}"
            )
            logger.info(f"   doc_id={result['doc_id']}")
            logger.info(f"   title={metadata['title'] or '(untitled)'}")
            logger.info(f"   author={metadata['author'] or '(unknown)'}")
            logger.info(f"   filepath={metadata['filepath']}")
            logger.info(
                f"   tokens={result['token_start']}:"
                f"{result['token_end']}"
            )
            logger.info(f"   passage_id={result['passage_id']}")
            logger.info(
                f"   {metadata['passage_text'] or '[Text unavailable]'}"
            )


if __name__ == "__main__":
    main()

"""
python src/passage_search_pipeline/passages_vector_search.py
Lance directory: D:\src\mutatis-mutandis\out\indexes\lance_albinism
Query: his head and hairs were white as white wool, and as snow, and his eyes were as a flame
Years: 1000-2000
Representation: mean4

1. year=1659 distance=0.0559
   doc_id=A45436
   title=A paraphrase and annotations upon all the books of the New Testament briefly explaining all the difficult places thereof / by H. Hammond.
   author=Hammond, Henry, 1605-1660.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A45436.P5.xml
   tokens=1269043:1269083
   passage_id=23e1f7f7515f6036cefbef06d71f32d671e2fedbc99a7690b4882921a5ab2e8a
   his head and his haires were whiteas it were a white fleece like wooll , as snow as white as snow ; and his eyes were as a flame of fire ; paraphrase . and his appearance from head to

2. year=1647 distance=0.0584
   doc_id=A57963
   title=Christ dying and drawing sinners to himself, or, A survey of our Saviour in his soule-suffering, his lovelynesse in his death, and the efficacie thereof in which some cases of soule-trouble in weeke beleevers ... are opened ... delivered in sermons on the Evangel according to S. John Chap. XII, vers. 27, 28, 29, 30, 31, 32, 33 ... / by Samuel Rutherford.
   author=Rutherford, Samuel, 1600?-1661.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57963.P5.xml
   tokens=162856:162896
   passage_id=9e18ea4e28a0f8bd87eb7d8a8bbe15a24e558747cc50c3b8b7a07c7b8b941957
   was clothed with a garment downe to the feet , and girt about the paps , with a golden girdle . uers . . his head and his haire were white like wooll , as white as snow , and

3. year=1647 distance=0.0590
   doc_id=A57963
   title=Christ dying and drawing sinners to himself, or, A survey of our Saviour in his soule-suffering, his lovelynesse in his death, and the efficacie thereof in which some cases of soule-trouble in weeke beleevers ... are opened ... delivered in sermons on the Evangel according to S. John Chap. XII, vers. 27, 28, 29, 30, 31, 32, 33 ... / by Samuel Rutherford.
   author=Rutherford, Samuel, 1600?-1661.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57963.P5.xml
   tokens=162876:162916
   passage_id=f1a3f146ff158ffe7b2584fe6fdda8539a0d6d962c0d9bdb73c5337af4ff1c3c
   . uers . . his head and his haire were white like wooll , as white as snow , and his eyes were as a flame of fire . vers . . and his feet like unto fine brasse ,

4. year=1655 distance=0.0601
   doc_id=A65188
   title=The retired mans meditations, or, The mysterie and power of godlines shining forth in the living Word to the unmasking the mysterie of iniquity in the most refined and purest forms : and withall presenting to view ... in which old light is restored and new light justified : being the witness which is given to this age / by Henry Vane.
   author=Vane, Henry, Sir, 1612?-1662.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A65188.P5.xml
   tokens=14296:14336
   passage_id=f81c99cd115adb00487b0a272e31190789bc0f0a3473f93ee8dc1c0d8a848dc3
   same throne , as rev . . whose garment was white as snow , and the hair of his head like the pure wool : his throne was like the fiery flame , and his wheels as burning fire :

5. year=1659 distance=0.0646
   doc_id=A43727
   title=Apokalypsis apokalypseos, or, The revelation revealed being a practical exposition on the revelation of St. John : whereunto is annexed a small essay, entituled Quinto-Monarchiæ, cum Quarto Omologia, or, A friendly complyance between Christ's monarchy, and the magistrates / by William Hicks ...
   author=Hicks, William, 1621-1660.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A43/A43727.P5.xml
   tokens=25554:25594
   passage_id=5a07412b7de7b2ad5172c8bdbf64d9b29100384bf314dd135a85a8869b745531
   there called the ancient of days , whose garment was white as snow , and the hair of his head as pure wool , his throne was like the fiery flame , and his wheels were as burning fire ?

6. year=1653 distance=0.0648
   doc_id=A62379
   title=The grand assises: or, The doctrine of the last generall judgment with the circumstances thereof: comprised and laid forth in a sermon preached at the assises holden for the county of Southampton at Winchester, on Wednesday, July 28, 1652. By William Sclater Doctor in Divinity, preacher of the word of God in Broadstreet, London.
   author=Sclater, William, 1609-1661.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A62/A62379.P5.xml
   tokens=1354:1394
   passage_id=2fe917020ffc1d6af06c2f96bba96c939433dea725cc4dd19ec22e98b04f086d
   upon his throne imperiall and paramount , whose garment was white as snow , and the hair of his head like pure wool , his throne was like the fiery flame , and his wheels like burning fire ; a

7. year=1665 distance=0.0659
   doc_id=A27214
   title=Some observations upon the apologie of Dr. Henry More for his mystery of godliness by J. Beaumont ...
   author=Beaumont, Joseph, 1616-1699.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A27214.P5.xml
   tokens=6377:6417
   passage_id=0d8b638aadebbb7f9b68524707062bdcb7608ad1c296e652434f1b83750990c4
   . . ) mentioneth his head and hair to be white as wooll or snow ; his eyes as a flame of fire ; his feet as burning brass ; his countenance as the sun shining in his strength .

8. year=1651 distance=0.0671
   doc_id=A81085
   title=The little horns doom & dovvnfall or A scripture-prophesie of King James, and King Charles, and of this present Parliament, unfolded. Wherein it appeares, that the late tragedies that have bin acted upon the scene of these three nations: and particularly, the late Kings doom and death, was so long ago, as by Daniel pred-eclared [sic]. And what the issue of all will be, is also discovered; which followes in the second part. By M. Cary, a servant of Jesus Christ.
   author=Cary, Mary.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A81/A81085.P5.xml
   tokens=6364:6404
   passage_id=491cd23a9799622a8e209f9ba12d03a9430391523c92ca5d407fcb05fba29a78
   the ancient of daies did sit , whose garment was white as snow , and the hair of his head like the pure wool , his throne like the fiery flame , and wheels as burning fire , a fiery

9. year=1655 distance=0.0679
   doc_id=A87004
   title=A discovery of the latitude of the loss of the earthly paradise by original sin. Occasioned by a disputation betwixt Mr. Matthias Rutton, min: of Boughton Munchalse in Kent, and the author hereof; branched out in these particulars, as followeth: I. First, an examination of his apologetical letter, with a full answer thereunto. II. Secondly, six arguments to prove that original sin exposed Adam and his posterity but to the first death. III. Thirdly, a discovery of the false and corrupted ministers by ten characters. IV. Fourthly, a discovery of the true ministers by ten characters. / By George Hammon, pastor to the Church of Christ meeting in Biddenden in Kent.
   author=Hammon, George.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A8/A87004.P5.xml
   tokens=26694:26734
   passage_id=761826517f6f7217632dd241dfd33b3d8cffe722e576989d72f73b928a3a6f41
   the antient of days did sit , whose garment was white as snow , and the hair of his head like unto pure wool , his throne was like the fiery flame , and his wheels like burning fire ;

10. year=1659 distance=0.0682
   doc_id=A80737
   title=Knovvledge & practice, or, a plain discourse of the chief things necessary to be known, believ'd, and practised in order to salvation. Drawn up, and principally intended for the use and benefit of North-Cadbury in Somersetshire, / by Samuel Cradock, B.D. & Pastor there: sometime fellow of Emmanuel Colledge in Cambridge.
   author=Cradock, Samuel, 1621?-1706.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A80/A80737.P5.xml
   tokens=168145:168185
   passage_id=33ff27da4296d857724f74483652b188498a8588034d9b5bbee6b306590c7cf7
   the ancient of daies did sit , whose garment was white as snow , and the hair of his head like the pure wool ; his throne was like the fiery flame , and his wheels as burning fire ,

11. year=1659 distance=0.0688
   doc_id=A45436
   title=A paraphrase and annotations upon all the books of the New Testament briefly explaining all the difficult places thereof / by H. Hammond.
   author=Hammond, Henry, 1605-1660.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A45436.P5.xml
   tokens=1269023:1269063
   passage_id=652df0787e57d938dfcb6ba15f06f17b0f5587c74801bc2da048c4f4e9be6289
   note on luk . . b . ) and this belt or girdle girt about the paps . , . his head and his haires were whiteas it were a white fleece like wooll , as snow as white as

12. year=1657 distance=0.0689
   doc_id=A30785
   title=The Jewish synagogue, or, An historical narration of the state of the Jewes at this day dispersed over the face of the whole earth ... / translated out of the learned Buxtorfius ... by A.B., Mr. A. of Q. Col. in Oxford.
   author=Buxtorf, Johann, 1599-1664.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A3/A30785.P5.xml
   tokens=109958:109998
   passage_id=79f32d07bbd42382d629d55c226489f4f63af3e5c71dfbd821e22b529c87f219
   the ancient of daies did sit , whose garment was white as snow , and the haire of his head like the pure wooll : his throne was like the fiery fl me , and his wheels as burning fire

13. year=1633 distance=0.0691
   doc_id=A08578
   title=An explanation of the generall Epistle of Saint Iude. Delivered in one and forty sermons, by that learned, reverend, and faithfull servant of Christ, Master Samuel Otes, parson of Sowthreps in Norfolke. Preached in the parish church of Northwalsham, in the same county, in a publike lecture. And now published for the benefit of Gods church, by Samuel Otes, his sonne, minister of the Word of God at Marsham
   author=Otes, Samuel, 1578 or 9-1658.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A08/A08578.P5.xml
   tokens=100521:100561
   passage_id=7189fb430745b23c31941888e43e842e45ee237cd75082c0322c14d316769aed
   the ancient of dayes did sit . whose garments was white as snow , and the hayre of his head like the pure wooll , his throne was like the fiery flame , and his wheeles as burning fire .

14. year=1649 distance=0.0697
   doc_id=A93110
   title=Of the foure last and greatest things: death, iudgement, heaven and hell. The description of the happinesse of heaven, and misery of hell, by way of antithesis. With the way or means to passe through death, and judgement, into heaven, and to avoid hell. / By VVilliam Shepheard, Esquire.
   author=Sheppard, William, d. 1675?
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A93/A93110.P5.xml
   tokens=53304:53344
   passage_id=da2c56df96681731f64105682dfbc2f35fe921bbf873d6d6108d68367d566071
   . . and he shall sit whose garment shall be white as snow , and the haire of his head like pure wool ; his throne like the fiery flame , and his wheels as burning fire ; a fiery

15. year=1645 distance=0.0702
   doc_id=A64611
   title=The summe of Christian religion, delivered by Zacharias Ursinus first, by way of catechism, and then afterwards more enlarged by a sound and judicious exposition, and application of the same : wherein also are debated and resolved the questions of whatsoever points of moment have been, or are controversed in divinitie / first Englished by D. Henry Parry, and now again conferred with the best and last Latine edition of D. David Pareus, sometimes Professour of Divinity in Heidelberge ; whereunto is added a large and full alphabeticall table of such matters as are therein contained ; together with all the Scriptures that are occasionally handled, by way either of controversie, exposition, or reconciliation, neither of which was done before, but now is performed for the readers delight and benefit ; to this work of Ursinus are now at last annexed the Theologicall miscellanies of D. David Pareus in which the orthodoxall tenets are briefly and solidly confirmed, and the contrary errours of the Papists, Ubiquitaries, Antitrinitaries, Eutychians, Socinians, and Arminians fully refuted ; and now translated into English out of the originall Latine copie by A.R.
   author=Ursinus, Zacharias, 1534-1583.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A64/A64611.P5.xml
   tokens=310050:310090
   passage_id=881e1118d71e496239b623d0d6ff6dffa9a14fbdb009ddde9e879411d79bdaed
   the ancient of dayes did sit , whose garment was white as snow , and the haire of his head like the pure wooll , vers . , . his throne was like the fiery flame , and his wheeles

16. year=1651 distance=0.0711
   doc_id=A86947
   title=The mystery of Christ in us, with the mystery of the Father, Word, and holy Ghost or Spirit, opened also, the parable of the rich man's flocks and herds, and the poor man's ewe-lamb, explicated. Likewise, the way that Christ takes to undo a man, and take away his life. Together with a discovery of the neerness of Christs coming, and of those glorious things which are to be fulfilled in these later days. Set forth and published by Ed. Hide jun.
   author=Hyde, Edward, 1607-1659.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A8/A86947.P5.xml
   tokens=28372:28412
   passage_id=7e42c807163adc2e58955c81526ec08b3a2ce802437d159010444bf1387a3dcb
   the ancient of days did sit , whose garment was white as snow , and the hair of his head like the pure wool ; his throne was like the very flame , and his wheeles as burning fire a

17. year=1634 distance=0.0719
   doc_id=A12093
   title=Mans last end the glorious vision and fruition of God. By Richard Sheldon Doctor in Divinity, one of his Maiesties chaplines
   author=Sheldon, Richard, d. 1642?
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A12/A12093.P5.xml
   tokens=27454:27494
   passage_id=4e4c897e98ed22d4c77e4b8f8ec08f7ec72648e4d6ef64d7e1a93ebe8c911822
   when his face did shyne as the sun , and his rayment was white as the light . s . mark , addes that his rayment became shyning , exceeding whyte as the snow , so as no fuller on

18. year=1644 distance=0.0730
   doc_id=A43318
   title=A sermon preached to the honourable House of Commons at their late solemne fast, Wednesday, December 27, 1643 by Alexander Henderson ...
   author=Henderson, Alexander, 1583?-1646.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A43318.P5.xml
   tokens=9076:9116
   passage_id=b943878f2bdfa1d10a2360782f7c6736bc7dd3957930a9120e09c160f7307ecf
   the antient of dayes did sit , whose garment was white as snow , and the hair of his head as pure wooll , his throne was like the fiery flame , and his wheels as burning fire . a

19. year=1650 distance=0.0735
   doc_id=A25250
   title=Ultima, = the last things in reference to the first and middle things: or certain meditations on life, death, judgement, hell, right purgatory, and heaven: delivered by Isaac Ambrose, minister of the Gospel at Preston in Amoundernes in Lancashire.
   author=Ambrose, Isaac, 1604-1664.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A25/A25250.P5.xml
   tokens=52437:52477
   passage_id=a26d6dcdec847abe0d505a4f628fafc04a12835440c519c5eb9105c034ddf86f
   and the ancient of days sate down , his garments white as snow , dan . . . and the hair of his head like pure wooll his throne like the fiery flame , and his wheeles as burning fire

20. year=1629 distance=0.0737
   doc_id=A16539
   title=The last battell of the soule in death diuided into eight cof̃erences ... : whereby are shown the diuerse skirmishes that are between the soule of man on his death-bedde, and the enemies of our saluation : carefullie digested for the comfort of the sicke  / by Mr. Zachary Boyd, preacher of Gods word at Glasgow.
   author=Boyd, Zacharie, 1585?-1653.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A1/A16539.P5.xml
   tokens=114540:114580
   passage_id=d72633331df6f9c48d101a9d595703254bcd209efceb8fd4b87d4447a37fb5f3
   the ancient of dayes did ite , base garn was white as snow , and the haire of his head lik the pure wool throne was like the , and his wheele as burning fire a fie e streame issued

21. year=1654 distance=0.0737
   doc_id=A79552
   title=Saint Chrysostome his Parænesis, or Admonition wherein hee recalls Theodorus the fallen. Or generally an exhortation for desperate sinners. / Translated by the Lord Viscount Grandison prisoner in the Tower.
   author=John Chrysostum, Saint, d. 407.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A7/A79552.P5.xml
   tokens=15376:15416
   passage_id=4b54c91f77f2f3f14e3dd047050c3b6691caa88040a3ed9f3df3b69d9c2cee49
   the antient of daies did sit , whose garment was white as snow , and the hair of his head like the pure wooll : his throne was a flame of fire , and his wheeles burning fire : a

22. year=1657 distance=0.0743
   doc_id=A47013
   title=Maran atha: or Dominus veniet Commentaries upon the articles of the Creed never heretofore printed. Viz. Of Christs session at the right hand of God and exaltation thereby. His being made Lord and Christ: of his coming to judge the quick and the dead. The resurredction of the body; and Life everlasting both in joy and torments. With divers sermons proper attendants upon the precedent tracts, and befitting these present times. By that holy man and profound divine, Thomas Jackson, D.D. President of Corpus Christi Coll. in Oxford.
   author=Jackson, Thomas, 1579-1640.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A47013.P5.xml
   tokens=79426:79466
   passage_id=5314baf28e14e696a292f5c8b4a9aaa1342acc1c3325fffbbda63eb2c9437fca
   the ancient of days did sit , whose garment was white as snow , and the hair of his head like pure wool ; his throne was like the fiery flame , and his wheels as burning fire . a

23. year=1656 distance=0.0745
   doc_id=A61453
   title=A plain and easie calculation of the name, mark, and number of the name of the beast ... humbly presented to the studious observers of Scripture-prophecies, God's works, and the times / by Nathaniel Stephens ... ; whereunto is prefixed, a commendatory epistle, written by Mr. Edm. Calamy.
   author=Stephens, Nathaniel, 1606?-1678.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A61453.P5.xml
   tokens=88745:88785
   passage_id=297a61ef5be243ec75e39c1eb6e8451e7f3c10179f075758c949a0ccdaf6d80c
   the thrones were cast down ; or more properly , i beheld till the thrones were set , and the ancient of days did sit , whose garment is as white as snow , and the hair of his head

24. year=1656 distance=0.0745
   doc_id=A61453
   title=A plain and easie calculation of the name, mark, and number of the name of the beast ... humbly presented to the studious observers of Scripture-prophecies, God's works, and the times / by Nathaniel Stephens ... ; whereunto is prefixed, a commendatory epistle, written by Mr. Edm. Calamy.
   author=Stephens, Nathaniel, 1606?-1678.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A61453.P5.xml
   tokens=88765:88805
   passage_id=04964231a85405063b9d88587af238fa5e4e5a266f8fb58600f439d1c46c2b4d
   ancient of days did sit , whose garment is as white as snow , and the hair of his head like the pure wooll , ch . . v . . in these words , why is god the father

25. year=1649 distance=0.0755
   doc_id=A78070
   title=The little Bible of the man or the Book of God opened in man by the power of the Lamb. Wherein God is the spirit or inside of the book, and man the letter or out-side of it. In whom, as in a glass, you may both behold the spirit and letter of the holy Scriptures in the new man; fulfilled and explained from Genesis to Jeremiah. This is the first volume of Gods Book in man. Written by a weak instrument of the Lords, Capt. T. Butler.
   author=Butler, Thomas, Captain.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A78/A78070.P5.xml
   tokens=31206:31246
   passage_id=2a2a9f6d9984f7de751366a3ba189f88e7e1ba9647a2f5bb04cea5e5064df944
   thou art fair , thou hast doves eyes ; a pure wisdom from above , and peaceable . secondly , thy hair is like a flock of goats , as white as snow , noting old age found in the

26. year=1653 distance=0.0756
   doc_id=A62379
   title=The grand assises: or, The doctrine of the last generall judgment with the circumstances thereof: comprised and laid forth in a sermon preached at the assises holden for the county of Southampton at Winchester, on Wednesday, July 28, 1652. By William Sclater Doctor in Divinity, preacher of the word of God in Broadstreet, London.
   author=Sclater, William, 1609-1661.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A62/A62379.P5.xml
   tokens=1334:1374
   passage_id=18c3b5c20e19b6c1f5c049cd93600e974c204b13648d307b2ac4548ae470fc4f
   was the daniel of the new testament ) is described under the title of the ancient of dayes , sitting upon his throne imperiall and paramount , whose garment was white as snow , and the hair of his head

27. year=1632 distance=0.0758
   doc_id=A12478
   title=An exposition of the Creed: or, An explanation of the articles of our Christian faith. Delivered in many afternoone sermons, by that reverend and worthy divine, Master Iohn Smith, late preacher of the Word at Clavering in Essex, and sometime fellow of Saint Iohns Colledge in Oxford. Now published for the benefit and behoofe of all good Christians, together with an exact table of all the chiefest doctrines and vses throughout the whole booke
   author=Smith, John, 1563-1616.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A1/A12478.P5.xml
   tokens=283082:283122
   passage_id=6f0ca998478716dc1be9563cc1479e471bdaf1d9da7eb3371792dee739270534
   devoure before him , and a mightie tempest shall be moved round about him . daniel . . and he ancient of dayes did fit , whose garment was as white as snow , and the haire of his head

28. year=1655 distance=0.0766
   doc_id=A74677
   title=Eugenius Theodidactus. The prophetical trumpeter sounding an allarum to England illustrating the fate of Great Britain, past, present, and to come. Such wonderful things to happen these seven yeers following, as have not been heard of heretofore. A celestial vision. VVith a description of heaven and heavenly things, motives to pacifie Gods threatned wrath: of a bloody, fiery way of the day of judgment, and of saints and angels. / Sung in a most heavenly hymn, to the great comfort of all good Christians, by the Muses most unworthy, John Heydon, gent. philomat.
   author=Heydon, John, b. 1629.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A7/A74677.P5.xml
   tokens=26147:26187
   passage_id=aaf8a89bba0bd7021559766c61ca73eca510682360944e3fdc73c8a28ffc9bf2
   least pimples . her lips like threds of scarlet , coral red , her temples faire , her hair like golden thred . her breath more savourie then mellifluous dew , her brests like two young twin roes white of

29. year=1644 distance=0.0776
   doc_id=A55917
   title=A commentary upon the divine Revelation of the apostle and evangelist, Iohn by David Pareus ... ; and specially some things upon the 20th chapter are observed by the same authour against the Millenaries ; translated out of the Latine into English, by Elias Arnold.
   author=Pareus, David, 1548-1622.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A55/A55917.P5.xml
   tokens=19578:19618
   passage_id=8f962367d4b671e4b134af761e70e45177ba7a3a005d0f0e2adb2869af7403f5
   christ towards the church , because the heart , which is the feat of love is between the brests . vers . : his head , and his haires were white like wool , as white as snow so the

30. year=1651 distance=0.0779
   doc_id=A81085
   title=The little horns doom & dovvnfall or A scripture-prophesie of King James, and King Charles, and of this present Parliament, unfolded. Wherein it appeares, that the late tragedies that have bin acted upon the scene of these three nations: and particularly, the late Kings doom and death, was so long ago, as by Daniel pred-eclared [sic]. And what the issue of all will be, is also discovered; which followes in the second part. By M. Cary, a servant of Jesus Christ.
   author=Cary, Mary.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A81/A81085.P5.xml
   tokens=6344:6384
   passage_id=cd37973b0f651e5e5784decb0f823128a30feae581167aff79b3fcc4c96adb84
   great day of judgement : for thus it runs , i beheld till the thrones were cast down , and the ancient of daies did sit , whose garment was white as snow , and the hair of his head
"""
