"""
passage_search_pipeline/passages_vector_search.py

Search using an existing passage vector found via `find_vector.py`:

   $ python src/passage_search_pipeline/find_doc_vectors.py albinism_case_report_psaila_1972

Yields:

   passage_id=ef2f4aaf68dc6c924a2f1218e470a555060be2cb886a0ef1ebf169b800508f8a
   # Passag text = "can be seen by contrast with the normally pigmented skin of the nurse holding her . her hair is fine , silky and whitish . her eyebrows and eyelashes are also white . she has gross photophobia and nystagmus ."

   python src/passage_search_pipeline/passages_vector_search.py \
   --vector-id ef2f4aaf68dc6c924a2f1218e470a555060be2cb886a0ef1ebf169b800508f8a \
   --limit 30

Or using a text query:

   python src/passage_search_pipeline/passages_vector_search.py \
   --query "white hair" \
   --limit 30

With negatives:

   python src/passage_search_pipeline/passages_vector_search.py \
   --query "white hair" \
   --representation l8 \
   --limit 50 \
   --negative-id dfbee39870fbbababa5fc24da247fb8468a82350bca1170209458b84568be2c3 \
   --negative-id afee38e3452766ffd28bc725ba733cec845fe8cb338c8ef93a52f850223fdbe7 \
   --beta 0.6

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


def load_passage_vector(db, passage_id, representation):
    """Load an existing passage vector from the selected MacBERTh representation."""
    prefix = f"passage_{representation}__macberth__"
    matches = []

    for table_name in db.list_tables().tables:
        if not table_name.startswith(prefix):
            continue

        table = db.open_table(table_name)
        escaped_id = passage_id.replace("'", "''")

        rows = (
            table.search()
            .where(f"passage_id = '{escaped_id}'")
            .limit(2)
            .to_list()
        )

        for row in rows:
            matches.append((table_name, row))

    if not matches:
        raise ValueError(
            f"No vector found for passage_id={passage_id!r}, "
            f"representation={representation!r}"
        )

    if len(matches) > 1:
        locations = [name for name, _ in matches]
        raise ValueError(
            f"Multiple vectors found for passage_id={passage_id!r}: "
            f"{locations}"
        )

    table_name, row = matches[0]
    vector = normalise_vector(row["vector"], "Stored passage vector")

    logger.info(
        "Loaded vector ID %s from %s (doc_id=%s, year=%s)",
        passage_id,
        table_name,
        row.get("doc_id"),
        row.get("pub_year"),
    )

    return vector


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
   parser.add_argument( "--vector-id", help="Use an existing passage vector by passage_id instead of encoding --query.", )
   parser.add_argument( "--start-year", type=int, default=0, help="Earliest publication year (default: 0).", )
   parser.add_argument( "--end-year", type=int, default=3000, help="Latest publication year (default: 3000).", )
   parser.add_argument( "--representation", default="l8", help="Passage representation (default: l8).", )
   parser.add_argument( "--limit", type=int, default=30, help="Maximum number of results (default: 30).", )
   parser.add_argument( "--negative-id", action="append", default=[], help="Passage ID of a negative example (can be repeated). " "These are averaged into a centroid used for margin re-ranking.", )
   parser.add_argument( "--beta", type=float, default=0.5, help="Weight of the negative centroid in re-ranking " "(score = sim_q - beta * sim_neg). Default: 0.5", )
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

   if args.vector_id:
      logger.info("Query vector ID: %s", args.vector_id)
      query_vector = load_passage_vector(
         db,
         args.vector_id,
         args.representation,
      )
   else:
      logger.info("Text query: %s", args.query)
      query_vector = normalise_vector(
         encode_passage_query(args.query, args.representation),
         "Text query",
      )

   negative_vectors = []
   if args.negative_id:
      for nid in args.negative_id:
         try:
               neg_vec = load_passage_vector(db, nid, args.representation)
               negative_vectors.append(neg_vec)
               logger.info("Loaded negative vector: %s", nid)
         except ValueError as e:
               logger.warning("Skipping negative %s: %s", nid, e)

      if not negative_vectors:
         logger.warning("No valid negative vectors loaded; proceeding without re-ranking")

   results = ps.search_passages(
      db=db,
      query_vector=query_vector,
      representation=args.representation,
      start_year=args.start_year,
      end_year=args.end_year,
      limit=args.limit,
      include_vectors=False,          # still False for the caller
      negative_vectors=negative_vectors or None,
      beta=args.beta,
   )


   if not results:
        logger.info("No results found.")
        return

   with get_connection(application_name="revelation-vector-test") as conn:
        for rank, result in enumerate(results, start=1):
            metadata = ps.fetch_passage_text(conn, result)
            logger.info("...............................................................")
            if "score" in result:
               logger.info(
                  f"{rank}. year={result['pub_year']} "
                  f"distance={result['distance']:.4f} "
                  f"score={result['score']:.4f}"
               )
            else:
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

Lance directory: D:\src\mutatis-mutandis\out\indexes\lance_albinism
Query: white hair
Years: 1000-2000
Representation:l8
Limit: 100
Text query: white hair
=============================================================
1. year=1657 distance=0.2432
   doc_id=A91851
   title=The universal body of physick in five books; comprehending the several treatises of nature, of diseases and their causes, of symptomes, of the preservation of health, and of cures. Written in Latine by that famous and learned doctor Laz. Riverius, counsellour and physician to the present King of France, and professor in the Vniversity of Montpelier. Exactly translated into English by VVilliam Carr practitioner in physick.
   author=Rivière, Lazare, 1589-1655.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A91/A91851.P5.xml
   tokens=72728:72768
   passage_id=6a39b1b7fab3fc593f4bc9e30f7bbb63f55b1694372562937b5ddc6effe1ad3b
   the colour of the same white . hair . hair soft and smooth , and from the beginning thin . second quality . yellow hair , because flegm by longer coction is so coloured . third figure . direct hairs
=============================================================
2. year=1657 distance=0.2487
   doc_id=A49255
   title=The penitent pardoned a treatise wherein is handled the duty of confession of sin and the priviledge of the pardon of sin : together with a discourse of Christs ascension into heaven and of his coming again from heaven : wherein the opinion of the Chiliasts is considered and solidly confuted / being the sum and substance of several sermons preached by that faithful servant of Christ, Mr. Christopher Love ...
   author=Love, Christopher, 1618-1651.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A49255.P5.xml
   tokens=29113:29153
   passage_id=ca265a6ea5242f3175b480761fe6bf4ebae500efc4848a753d606afe11389598
   white in the sore , they note to be continuance in sin , living from youth to old age in sin , till the hair be white and gray . secondly , there was to be quick and raw flesh
=============================================================
3. year=1655 distance=0.2498
   doc_id=A57667
   title=Pansebeia, or, A view of all religions in the world with the severall church-governments from the creation, to these times : also, a discovery of all known heresies in all ages and places, and choice observations and reflections throughout the whole / by Alexander Ross.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57667.P5.xml
   tokens=113440:113480
   passage_id=0e1861eb9b1f1a969672ea850941a03d64808fce646459c8a48febb2b2a99f6e
   a black corner ' d cap . but saint austins eremites wear a black coat , with a hood of the same colour ; underneath there is a white little coat . their girdle is of leather with a buckle
=============================================================
4. year=1657 distance=0.2561
   doc_id=A91851
   title=The universal body of physick in five books; comprehending the several treatises of nature, of diseases and their causes, of symptomes, of the preservation of health, and of cures. Written in Latine by that famous and learned doctor Laz. Riverius, counsellour and physician to the present King of France, and professor in the Vniversity of Montpelier. Exactly translated into English by VVilliam Carr practitioner in physick.
   author=Rivière, Lazare, 1589-1655.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A91/A91851.P5.xml
   tokens=72708:72748
   passage_id=668b61de857ee14e93d3aeae92b56260d850bca09d15bd1f51783544c665abb5
   feet chiefly and hands very cold in winter . qualities second . a soft and smooth skin . third . the colour of the same white . hair . hair soft and smooth , and from the beginning thin .
=============================================================
5. year=1653 distance=0.2562
   doc_id=A88592
   title=The souls cordiall in two treatises. I. Teaching how to be eased of the guilt of sin. II. Discovering advantages by Christs ascension. The third volum. / By that faithfull labourer in the Lords vineyard Mr. Christopher Love, pastor of Lawrence Jury, London.
   author=Love, Christopher, 1618-1651.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A88/A88592.P5.xml
   tokens=28311:28351
   passage_id=0c59d1bedcf399076ada719c11154f8aef366976bbf7ec95eeed6c16882189ab
   hair white in the sore , they note to be continuance in sin , living from youth to old age in sin , till the hair be white and gray . secondly , there was to be quick and raw
=============================================================
6. year=1655 distance=0.2574
   doc_id=A57667
   title=Pansebeia, or, A view of all religions in the world with the severall church-governments from the creation, to these times : also, a discovery of all known heresies in all ages and places, and choice observations and reflections throughout the whole / by Alexander Ross.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57667.P5.xml
   tokens=201508:201548
   passage_id=c800fb5fc678c67fe08481a324423903ed5cf62b3fb8c7836069a8c1cf8be1d3
   because it is said cant . . my beloved is white and red . vvhite in his confessors and virgins , red in his martyrs ; these are the roses and lillies of the valley . black is worn upon
=============================================================
7. year=1850 distance=0.2575
   doc_id=CLMET3208
   title=Olive
   author=Craik, Dinah Maria Mulock
   filepath=clmet/corpus/txt/plain/CLMET3_1_3_208.txt
   tokens=182716:182756
   passage_id=741b4531b836591f27bdaf0e53a07ea2ede35291ff233e0fdd8f803f1f474136
   Christal Manners . She wore a garb half - secular , half - religious . Her black serge dress betrayed no attention to fashion , scarcely even to neatness ; her beautiful hair was all put back under a white
=============================================================
8. year=1686 distance=0.2598
   doc_id=A28396
   title=The gentlemans recreation in two parts : the first being an encyclopedy of the arts and sciences ... the second part treats of horsmanship, hawking, hunting, fowling, fishing, and agriculture : with a short treatise of cock-fighting ... : all which are collected from the most authentick authors, and the many gross errors therein corrected, with great enlargements ... : and for the better explanation thereof, great variety of useful sculptures, as nets, traps, engines, &c. are added for the taking of beasts, fowl and fish : not hitherto published by any : the whole illustrated with about an hundred ornamental and useful sculptures engraven in copper, relating to the several subjects.
   author=Blome, Richard, d. 1705.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28396.P5.xml
   tokens=475348:475388
   passage_id=dfbee39870fbbababa5fc24da247fb8468a82350bca1170209458b84568be2c3
   august , and about the end of september ; the blush belgick rose , or the white province as being of a pale blush . thus much for red roses . the single yellow rose , by some florists esteemed
=============================================================
9. year=1661 distance=0.2598
   doc_id=A28194
   title=Ostenta Carolina, or, The late calamities of England with the authors of them the great happiness and happy government of K. Charles II ensuing, miraculously foreshewn by the finger of God in two wonderful diseases, the rekets and kings-evil : wherein is also shewen and proved (I.) that the rekets after a while shall seize on no more children but quite vanish through the mercy of God and by means of K. Charles II., (II.) that K. Charles II is the last of kings which shall so heal the kings-evil / discovered by the hand of the Lord upon his unworthy servant and His Majesties subject, John Bird ...
   author=Bird, John, 17th cent.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28194.P5.xml
   tokens=12269:12309
   passage_id=73636692a75e54dfba43069432b3f4aaf0972ab437f47adc3213e9992621b3d6
   reddish sore in a bald head or forehead . in garments if there did appear a greenish or reddish spot and spreading . the leprosie in a house were hollow strakes greenish or reddish , appearing deeper then the wall
=============================================================
10. year=1836 distance=0.2601
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=1243:1283
   passage_id=0dcd690db5ae073ae03dc831d583a9b5f2008805182b718685e51733272d6419
   who are termed white negroes , have coarse woolly hair of a white colour . the red iris has often a tremulous motion , and the eye is very sensible to light . the skin in european albinos is like
=============================================================
11. year=1686 distance=0.2602
   doc_id=A28396
   title=The gentlemans recreation in two parts : the first being an encyclopedy of the arts and sciences ... the second part treats of horsmanship, hawking, hunting, fowling, fishing, and agriculture : with a short treatise of cock-fighting ... : all which are collected from the most authentick authors, and the many gross errors therein corrected, with great enlargements ... : and for the better explanation thereof, great variety of useful sculptures, as nets, traps, engines, &c. are added for the taking of beasts, fowl and fish : not hitherto published by any : the whole illustrated with about an hundred ornamental and useful sculptures engraven in copper, relating to the several subjects.
   author=Blome, Richard, d. 1705.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28396.P5.xml
   tokens=475368:475408
   passage_id=afee38e3452766ffd28bc725ba733cec845fe8cb338c8ef93a52f850223fdbe7
   of a pale blush . thus much for red roses . the single yellow rose , by some florists esteemed but a wild rose ; the scarlet rose of austria groweth much like the former , with single flowers ,
=============================================================
12. year=1661 distance=0.2603
   doc_id=A28194
   title=Ostenta Carolina, or, The late calamities of England with the authors of them the great happiness and happy government of K. Charles II ensuing, miraculously foreshewn by the finger of God in two wonderful diseases, the rekets and kings-evil : wherein is also shewen and proved (I.) that the rekets after a while shall seize on no more children but quite vanish through the mercy of God and by means of K. Charles II., (II.) that K. Charles II is the last of kings which shall so heal the kings-evil / discovered by the hand of the Lord upon his unworthy servant and His Majesties subject, John Bird ...
   author=Bird, John, 17th cent.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28194.P5.xml
   tokens=12278:12318
   passage_id=accd7da486ab2be226b013a298772e0fd6d343218d091a78114a42ae0845338c
   in garments if there did appear a greenish or reddish spot and spreading . the leprosie in a house were hollow strakes greenish or reddish , appearing deeper then the wall , and spreading . see of all thls ,
=============================================================
13. year=1686 distance=0.2612
   doc_id=A28396
   title=The gentlemans recreation in two parts : the first being an encyclopedy of the arts and sciences ... the second part treats of horsmanship, hawking, hunting, fowling, fishing, and agriculture : with a short treatise of cock-fighting ... : all which are collected from the most authentick authors, and the many gross errors therein corrected, with great enlargements ... : and for the better explanation thereof, great variety of useful sculptures, as nets, traps, engines, &c. are added for the taking of beasts, fowl and fish : not hitherto published by any : the whole illustrated with about an hundred ornamental and useful sculptures engraven in copper, relating to the several subjects.
   author=Blome, Richard, d. 1705.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28396.P5.xml
   tokens=475374:475414
   passage_id=0bd2b090023cd7e8ebcd3d4b3a9d24c108c63670058bdc8240913d894c8a02aa
   much for red roses . the single yellow rose , by some florists esteemed but a wild rose ; the scarlet rose of austria groweth much like the former , with single flowers , which within are scarlet , and
=============================================================
14. year=1657 distance=0.2620
   doc_id=A53713
   title=Of communion with God the Father, Sonne, and Holy Ghost, each person distinctly in love, grace, and consolation, or, The saints fellowship with the Father, Sonne, and Holy Ghost, unfolded by John Owen ...
   author=Owen, John, 1616-1683.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A53713.P5.xml
   tokens=26986:27026
   passage_id=96e7039335c9d1913bbc3636af821dc82891e61ec8ef6cf197670ff36244a38f
   said , his garment was white as snow , and the haire of his head as pure wooll . and of christ in his transfiguration , when he had on him a mighty lustre of the deity , his face
=============================================================
15. year=1657 distance=0.2621
   doc_id=A49255
   title=The penitent pardoned a treatise wherein is handled the duty of confession of sin and the priviledge of the pardon of sin : together with a discourse of Christs ascension into heaven and of his coming again from heaven : wherein the opinion of the Chiliasts is considered and solidly confuted / being the sum and substance of several sermons preached by that faithful servant of Christ, Mr. Christopher Love ...
   author=Love, Christopher, 1618-1651.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A49255.P5.xml
   tokens=29133:29173
   passage_id=d3eabaf999f7424bb12d39d141e336d8469b13994c5cbd1d6bbdc9f5f9784ef2
   sin , till the hair be white and gray . secondly , there was to be quick and raw flesh in the rising , they understand it to be adventurousnesse in a sinner to commit sin against a raw ,
=============================================================
16. year=1972 distance=0.2623
   doc_id=albinism_case_report_psaila_1972
   title=Albinism – A Case Report
   author=A.J. Psaila M.D., D.C.H. (Lond.), M.R.C.P. (U.K.), M.R.C.P. (Edin.)
   filepath=misc/albinism_case_report_psaila_1972.xml
   tokens=268:308
   passage_id=3676fe665ff366c98f0b0719959ba1af686dea07c352bd54a7d74b9856e273a1
   pigmented skin of the nurse holding her . her hair is fine , silky and whitish . her eyebrows and eyelashes are also white . she has gross photophobia and nystagmus . in sunlight or bright artificial light , her
=============================================================
17. year=1655 distance=0.2624
   doc_id=A57667
   title=Pansebeia, or, A view of all religions in the world with the severall church-governments from the creation, to these times : also, a discovery of all known heresies in all ages and places, and choice observations and reflections throughout the whole / by Alexander Ross.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57667.P5.xml
   tokens=201493:201533
   passage_id=523a04be0288abe34f01998fb8b9475a701c8024da8ad71bd47006ad389a5903
   fire : in some places white is worn on the festivities of the martyrs , because it is said cant . . my beloved is white and red . vvhite in his confessors and virgins , red in his martyrs
=============================================================
18. year=1835 distance=0.2631
   doc_id=CLMET3157
   title=King O'Neil
   author=Gore, Catherine Grace Frances
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_157.txt
   tokens=205:245
   passage_id=ca58e4fbe98d457880df0b90afdd78f10cda9e7ed1a505d35fc1aa426099505d
   poplin coat , embroidered waistcoat , white silk breeches , three - cornered hat trimmed with white feather ) Countess Dillon ( Pink and silver brocade open dress , white satin petticoat flounced , and lace ruffles ) Marchioness de
=============================================================
19. year=1655 distance=0.2631
   doc_id=A57667
   title=Pansebeia, or, A view of all religions in the world with the severall church-governments from the creation, to these times : also, a discovery of all known heresies in all ages and places, and choice observations and reflections throughout the whole / by Alexander Ross.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57667.P5.xml
   tokens=113460:113500
   passage_id=24a017437490a9207b71240fb3b9e48fb8a40cd7b7f02c10c5d09549bdaedff7
   the same colour ; underneath there is a white little coat . their girdle is of leather with a buckle of horn . after the example of these monastical canons , there were ecclesiastical canons erected , who instead of
=============================================================
20. year=1661 distance=0.2635
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=887093:887133
   passage_id=36947521c15630d76a3f45ebf63d96160ac710683302f7c9745fd793818e64f3
   and white , and shining . and vers . . fine linnen white and clean . see ch . . . pure myrrhe ; ; that holy and heavenly word , which droppeth ( as myrrhe ) out of the
=============================================================
21. year=1655 distance=0.2643
   doc_id=A57667
   title=Pansebeia, or, A view of all religions in the world with the severall church-governments from the creation, to these times : also, a discovery of all known heresies in all ages and places, and choice observations and reflections throughout the whole / by Alexander Ross.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57667.P5.xml
   tokens=113420:113460
   passage_id=b3564f34ae35d39ed40981d615a87603148f52a89c09451e047364f4207245b2
   crowns are shaven like other friers ; and when they go abroad , they wear a bread hat , or a black corner ' d cap . but saint austins eremites wear a black coat , with a hood of
=============================================================
22. year=1836 distance=0.2645
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=1903:1943
   passage_id=b6d34671ac92c01287eec5ff7ba582027ba994a40515fb2e3815bfc89b71cd44
   like the nose of a white horse ; with white hair , beard , eye brows and eye lashes ; red tender eyes , a short sight , and scurfy skins , covered with a kind of white down .
=============================================================
23. year=1656 distance=0.2648
   doc_id=A84087
   title=Pearls of eloquence, or, The school of complements Wherein ladies, gentlewomen, and schollars, may accommodate their courtly practice with gentile ceremonies, complemental, amorous, and high expressions of speaking, or writing of letters. By VV. Elder, Gent.
   author=Elder, William, fl. 1680-1700.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A84/A84087.P5.xml
   tokens=30413:30453
   passage_id=f85c78acf23f5142d2d63b0e18298e74bd987f50a63b4e5ea5d2853513d128dc
   shine in a shady place . on the colour of her face . such colour hath her face , as when the sun , in summer his first rising hath begun on her eye brows and ch ks . each
=============================================================
24. year=1661 distance=0.2653
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=887113:887153
   passage_id=b70e1274f9bbb3b33143bddd0bff21e1dde5b38283583896608a9d556f1515cb
   . pure myrrhe ; ; that holy and heavenly word , which droppeth ( as myrrhe ) out of the mouth of christ , cant . . . his lips like lillies , dropping down pure myrrhe . pure water
=============================================================
25. year=1686 distance=0.2658
   doc_id=A28396
   title=The gentlemans recreation in two parts : the first being an encyclopedy of the arts and sciences ... the second part treats of horsmanship, hawking, hunting, fowling, fishing, and agriculture : with a short treatise of cock-fighting ... : all which are collected from the most authentick authors, and the many gross errors therein corrected, with great enlargements ... : and for the better explanation thereof, great variety of useful sculptures, as nets, traps, engines, &c. are added for the taking of beasts, fowl and fish : not hitherto published by any : the whole illustrated with about an hundred ornamental and useful sculptures engraven in copper, relating to the several subjects.
   author=Blome, Richard, d. 1705.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28396.P5.xml
   tokens=503263:503303
   passage_id=8f3aa9086ed20c4c54dfea55691563ea4d740c61433410a0542c79c63cd8e867
   wool easie to pull off ; and when he is dead , his belly will be full of water , his fat yellow , his liver putrified , and his flesh moist and waterish . of sickness in general ,
=============================================================
26. year=1820 distance=0.2659
   doc_id=CLMET3149
   title=Prometheus unbound
   author=Shelley, Percy Bysshe
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_149.txt
   tokens=24520:24560
   passage_id=93cfc562d97132f2d8395652cd28ba78065c8d2ed83c281215a596d1db842d88
   flowing folds Of its white robe , woof of ethereal pearl . Its hair is white , the brightness of white light Scattered in strings ; yet its two eyes are heavens 225 Of liquid darkness , which the Deity
=============================================================
27. year=1653 distance=0.2660
   doc_id=A88592
   title=The souls cordiall in two treatises. I. Teaching how to be eased of the guilt of sin. II. Discovering advantages by Christs ascension. The third volum. / By that faithfull labourer in the Lords vineyard Mr. Christopher Love, pastor of Lawrence Jury, London.
   author=Love, Christopher, 1618-1651.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A88/A88592.P5.xml
   tokens=28331:28371
   passage_id=6f3bf51b5efb02445f4c8ba0d0635e523fc7951b68a62c5961e8809e6117527a
   in sin , till the hair be white and gray . secondly , there was to be quick and raw flesh in the rising , they understand it to be adventurousnesse in a sinner to commit sin against a raw
=============================================================
28. year=1836 distance=0.2662
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=3972:4012
   passage_id=4d409d3de273c8a309f871e2631ee9446e57f90b4bd3290fb2898a6514746f95
   xanthous or intermediate white negroes with red , yellow or flaxen hair , blue grey brown eyes , and fair or ruddy skins follow in the original text . section iiof the structure of the parts on which the variety
=============================================================
29. year=1923 distance=0.2662
   doc_id=toomer-cane
   title=Cane
   author=Jean Toomer
   filepath=misc/literature-in-context/Toomer/toomer-cane.xml
   tokens=6435:6475
   passage_id=365d4f1ce079f49fa8b732a2e0358d4eaf0eab33603d3eecbaeffdec4c6cb462
   resplendently she gleams , cloine dreams , lips pressed against my heart . esther nine . esther ' s hair falls in soft curls about her high cheek boned chalk white face . esther ' s hair would be beautiful
=============================================================
30. year=1640 distance=0.2662
   doc_id=A04984
   title=An easy method to know the causes and signs of the humour most ruleth in the body and to avoid thereby things hurtful: as also to enable men to give better account of the state of their bodies, when they are diseased to the physitian, and not hazard the lives of themselves and their freinds by only urging him to prophesy (and that often falsily) the disease by onely the inspection of vrine.
   author=La Framboisière, Nicolas Abraham de, b. 16th cent.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A04/A04984.P5.xml
   tokens=2483:2523
   passage_id=a04a9420f5b0ddfab13b904b536fcd0e0112f3313b89eb80b23d7f5e2f7ce056
   urine is white or pale , now thin , now thick , and troubled with much setling . phlegme voluntarily breaking forth by vomit or dejection , and a body overflowing with moisture . white flux in women . the
=============================================================
31. year=1835 distance=0.2663
   doc_id=CLMET3157
   title=King O'Neil
   author=Gore, Catherine Grace Frances
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_157.txt
   tokens=145:185
   passage_id=5d1bdac3fd75ed9b04aa58426134e64ef798e487bfc5fc6e24395867d3962691
   coat trimmed with gold lace , large yellow cuffs , white breeches , high black boots , hat trimmed with feather , and sword ) Major Macnamara ( Dressed as Major de Burgh ) Cornet O ' Callaghan ( Dressed
=============================================================
32. year=1836 distance=0.2664
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=473:513
   passage_id=9ccc436a934cedafd99f087ce4c6f5b367ecb16333ee2e6b458c50ae311fcdfd
   to become a person of the ordinary fair complexion with blue eyes . the preceding remarks indicate the propriety of dividing the complexions of mankind into three classes , distinguished by the colour of the eyes and hair . this
=============================================================
33. year=1686 distance=0.2666
   doc_id=A28396
   title=The gentlemans recreation in two parts : the first being an encyclopedy of the arts and sciences ... the second part treats of horsmanship, hawking, hunting, fowling, fishing, and agriculture : with a short treatise of cock-fighting ... : all which are collected from the most authentick authors, and the many gross errors therein corrected, with great enlargements ... : and for the better explanation thereof, great variety of useful sculptures, as nets, traps, engines, &c. are added for the taking of beasts, fowl and fish : not hitherto published by any : the whole illustrated with about an hundred ornamental and useful sculptures engraven in copper, relating to the several subjects.
   author=Blome, Richard, d. 1705.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A2/A28396.P5.xml
   tokens=503261:503301
   passage_id=0343b222d7eaebbdb3489f055bc3da3b77f860de08253a9c48d1882898b06582
   and his wool easie to pull off ; and when he is dead , his belly will be full of water , his fat yellow , his liver putrified , and his flesh moist and waterish . of sickness in
=============================================================
34. year=1657 distance=0.2666
   doc_id=A91851
   title=The universal body of physick in five books; comprehending the several treatises of nature, of diseases and their causes, of symptomes, of the preservation of health, and of cures. Written in Latine by that famous and learned doctor Laz. Riverius, counsellour and physician to the present King of France, and professor in the Vniversity of Montpelier. Exactly translated into English by VVilliam Carr practitioner in physick.
   author=Rivière, Lazare, 1589-1655.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A91/A91851.P5.xml
   tokens=72743:72783
   passage_id=5769390852cf5a9d4e2f3cefa0a9ae4997cee99eeb2b5c2d87fc07ba9b8e00f6
   from the beginning thin . second quality . yellow hair , because flegm by longer coction is so coloured . third figure . direct hairs , because the skin being void of dryness , the passages in it are easie
=============================================================
35. year=1972 distance=0.2669
   doc_id=albinism_case_report_psaila_1972
   title=Albinism – A Case Report
   author=A.J. Psaila M.D., D.C.H. (Lond.), M.R.C.P. (U.K.), M.R.C.P. (Edin.)
   filepath=misc/albinism_case_report_psaila_1972.xml
   tokens=260:300
   passage_id=ef2f4aaf68dc6c924a2f1218e470a555060be2cb886a0ef1ebf169b800508f8a
   can be seen by contrast with the normally pigmented skin of the nurse holding her . her hair is fine , silky and whitish . her eyebrows and eyelashes are also white . she has gross photophobia and nystagmus .
=============================================================
36. year=1836 distance=0.2674
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=1223:1263
   passage_id=0cd69d0463e770297a968d54cf202e8ed024eaa2b67878ce15073517b9c1d94e
   seen examples of both , in persons born in europe , of genuine european descent . those african albinos , who are termed white negroes , have coarse woolly hair of a white colour . the red iris has often
=============================================================
37. year=1661 distance=0.2675
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=948056:948096
   passage_id=39a4ac2525732d2751f86a97caf72af6d763e69f957e41f8a56fec53a5e00399
   himself without spot of sin : red , his sinfull case by imputation , cor . . . and so these two colours are used to signifie sinfulnesse and innocency , isa . . . and red betokeneth his sufferings
=============================================================
38. year=1835 distance=0.2678
   doc_id=CLMET3157
   title=King O'Neil
   author=Gore, Catherine Grace Frances
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_157.txt
   tokens=125:165
   passage_id=b18fa10570b4402f240375c0f45fe9d179493486ab8abe32f285f112b7ff8126
   each shoulder , white breeches , silk stockings , and shoes ) Major de Burgh ( Scarlet broad - skirted coat trimmed with gold lace , large yellow cuffs , white breeches , high black boots , hat trimmed with
=============================================================
39. year=1836 distance=0.2683
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=263:303
   passage_id=3a999312c8866d4bb8f2b02ab9d269428e4d81ea1d5f148b103077f7b1b5dd3b
   syria and of barbary are often very white , though they have black hair , but this is the result of careful protection from the sun . and in europe , black haired women are termed brunettes , from the
=============================================================
40. year=1661 distance=0.2686
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=671052:671092
   passage_id=18f7e6b2d911dbbb0a46f6a0f63b0067ef954309f09f26927a096736befaf1f8
   bright ( or white ) linnen . he alludeth herein to the cloathing of the priests under the law , exod . . . some understand by this pure linnen , the equity and integrity of gods judgements upon his
=============================================================
41. year=1847 distance=0.2686
   doc_id=bronte-jane-eyre
   title=Jane Eyre: An Autobiography
   author=Charlotte Bronte
   filepath=misc/literature-in-context/BronteC/bronte-jane-eyre.xml
   tokens=48089:48129
   passage_id=e21ddac2adffc588aebe32381f51420d715aebe90668491a3061228eddb6f43d
   rich contrast crimson couches and ottomans ; while the ornaments on the pale parian mantelpiece were of sparkling bohemian glass , ruby red ; and between the windows large mirrors repeated the general blending of snow and fire . in
=============================================================
42. year=1640 distance=0.2687
   doc_id=A04984
   title=An easy method to know the causes and signs of the humour most ruleth in the body and to avoid thereby things hurtful: as also to enable men to give better account of the state of their bodies, when they are diseased to the physitian, and not hazard the lives of themselves and their freinds by only urging him to prophesy (and that often falsily) the disease by onely the inspection of vrine.
   author=La Framboisière, Nicolas Abraham de, b. 16th cent.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A04/A04984.P5.xml
   tokens=2443:2483
   passage_id=3cc3fc20fd947fd4af82f3a28f2e35f9ad3b2dd9aa5f5271daada778efd4ce52
   and heavy . dreames filled with water , raine , snow and drowning . heaviness of the head . the pulse small , slow , rare , soft . slow concoction , sowre belshings , loathing after eating . the
=============================================================
43. year=1972 distance=0.2687
   doc_id=albinism_case_report_psaila_1972
   title=Albinism – A Case Report
   author=A.J. Psaila M.D., D.C.H. (Lond.), M.R.C.P. (U.K.), M.R.C.P. (Edin.)
   filepath=misc/albinism_case_report_psaila_1972.xml
   tokens=240:280
   passage_id=6364900f793649d9ce6d43d24e59e60c218bf9548b14100b9340c04b503e490a
   photograph shows some of the features of this child ' s pigment abnormality . her skin is extremely white which can be seen by contrast with the normally pigmented skin of the nurse holding her . her hair is fine
=============================================================
44. year=1972 distance=0.2688
   doc_id=albinism_case_report_psaila_1972
   title=Albinism – A Case Report
   author=A.J. Psaila M.D., D.C.H. (Lond.), M.R.C.P. (U.K.), M.R.C.P. (Edin.)
   filepath=misc/albinism_case_report_psaila_1972.xml
   tokens=912:952
   passage_id=9d460172d01cac7df8aa3686b3a63f078b76db9fe41487599b57e9351ad54ae0
   . ) . b ) partial albinism : is more common . here the hair is yellow or light brown . errors of refraction , photophobia and nystagmus are common . the iris is light blue or pink . the
=============================================================
45. year=1835 distance=0.2688
   doc_id=CLMET3157
   title=King O'Neil
   author=Gore, Catherine Grace Frances
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_157.txt
   tokens=221:261
   passage_id=cee2effcbd50d8c9aa58f9bbbae979809dcfca9405cb7e248d90e52af49d0aa6
   white feather ) Countess Dillon ( Pink and silver brocade open dress , white satin petticoat flounced , and lace ruffles ) Marchioness de Clermont ( White satin open dress , petticoat with flounce , and ruffles ) Duchess de
=============================================================
46. year=1661 distance=0.2693
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=887115:887155
   passage_id=78b3154b3d9dd5e535219e5fcb9e76794a6465b9f17acb26c4c11af61bd5006f
   myrrhe ; ; that holy and heavenly word , which droppeth ( as myrrhe ) out of the mouth of christ , cant . . . his lips like lillies , dropping down pure myrrhe . pure water ; ;
=============================================================
47. year=1640 distance=0.2695
   doc_id=A04984
   title=An easy method to know the causes and signs of the humour most ruleth in the body and to avoid thereby things hurtful: as also to enable men to give better account of the state of their bodies, when they are diseased to the physitian, and not hazard the lives of themselves and their freinds by only urging him to prophesy (and that often falsily) the disease by onely the inspection of vrine.
   author=La Framboisière, Nicolas Abraham de, b. 16th cent.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A04/A04984.P5.xml
   tokens=2463:2503
   passage_id=3b826d5c8eba6b2adcb36cd18fbed7b9bee584bbe44d267adb4ef74af7961887
   pulse small , slow , rare , soft . slow concoction , sowre belshings , loathing after eating . the urine is white or pale , now thin , now thick , and troubled with much setling . phlegme voluntarily
=============================================================
48. year=1850 distance=0.2696
   doc_id=CLMET3208
   title=Olive
   author=Craik, Dinah Maria Mulock
   filepath=clmet/corpus/txt/plain/CLMET3_1_3_208.txt
   tokens=182736:182776
   passage_id=4be507d1f16798640818af01bf5746ee91cd4543e5733f572d3c561f23f460cf
   no attention to fashion , scarcely even to neatness ; her beautiful hair was all put back under a white linen veil , and her whole appearance showed that last bitter change in a woman ' s nature , when
=============================================================
49. year=1847 distance=0.2697
   doc_id=bronte-jane-eyre
   title=Jane Eyre: An Autobiography
   author=Charlotte Bronte
   filepath=misc/literature-in-context/BronteC/bronte-jane-eyre.xml
   tokens=48082:48122
   passage_id=5ec81c3ee9e671ae5abf8955820c02894e9e6f27098a201bd7ae6f1881161661
   vine leaves , beneath which glowed in rich contrast crimson couches and ottomans ; while the ornaments on the pale parian mantelpiece were of sparkling bohemian glass , ruby red ; and between the windows large mirrors repeated the general
=============================================================
50. year=1652 distance=0.2698
   doc_id=A57647
   title=Arcana microcosmi, or, The hid secrets of man's body discovered in an anatomical duel between Aristotle and Galen concerning the parts thereof : as also, by a discovery of the strange and marveilous diseases, symptomes & accidents of man's body : with a refutation of Doctor Brown's Vulgar errors, the Lord Bacon's natural history, and Doctor Harvy's book, De generatione, Comenius, and others : whereto is annexed a letter from Doctor Pr. to the author, and his answer thereto, touching Doctor Harvy's book De Generatione / by A.R.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57647.P5.xml
   tokens=62715:62755
   passage_id=d9aa8cb35e55263046cd80306dbebc3e2ceb9206942f3211b62d79526116961a
   hard , and blackneth soft things ; so he makes the ethiopians teeth white , the skin black ; he makes the green corn turn white and hard with his heat , and at the same time makes the plumb
=============================================================
51. year=1836 distance=0.2705
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=2509:2549
   passage_id=9fa19e534c852df59030330ddc95376e756a12d541f14bce92f429fedd1d4875
   but one sister was white like himself . his skin , from exposure to the sun , had acquired a slight reddish tinge , and was covered with a great number of black or brown spots , like freckles ,
=============================================================
52. year=1640 distance=0.2709
   doc_id=A09011
   title=Theatrum botanicum: = The theater of plants. Or, An herball of a large extent containing therein a more ample and exact history and declaration of the physicall herbs and plants that are in other authours, encreased by the accesse of many hundreds of new, rare, and strange plants from all the parts of the world, with sundry gummes, and other physicall materials, than hath beene hitherto published by any before; and a most large demonstration of their natures and vertues. Shevving vvithall the many errors, differences, and oversights of sundry authors that have formerly written of them; and a certaine confidence, or most probable conjecture of the true and genuine herbes and plants. Distributed into sundry classes or tribes, for the more easie knowledge of the many herbes of one nature and property, with the chiefe notes of Dr. Lobel, Dr. Bonham, and others inserted therein. Collected by the many yeares travaile, industry, and experience in this subject, by Iohn Parkinson apothecary of London, and the Kings herbarist. And published by the Kings Majestyes especial
   author=Parkinson, John, 1567-1650.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A09/A09011.P5.xml
   tokens=181593:181633
   passage_id=336a9dc0426aa22cc4fb26a00a48db76fef1eb6ff17640691c4ef93631fec78a
   of alypum : the flowers are white and small , set close together , in a long tuft , but so covered with the white woollinesse that they can scarse bee perceived : after which come small blacke seede ,
=============================================================
53. year=1661 distance=0.2709
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=671065:671105
   passage_id=1295284c2d14ae71f849146a7d9961f134dd2324d1ce399366e2db86ba2623eb
   of the priests under the law , exod . . . some understand by this pure linnen , the equity and integrity of gods judgements upon his enemies ; junius writeth , that this kind of cloathing with fine pure
=============================================================
54. year=1658 distance=0.2710
   doc_id=A47663
   title=The secret miracles of nature in four books : learnedly and moderately treating of generation, and the parts thereof, the soul, and its immortality, of plants and living creatures, of diseases, their symptoms and cures, and many other rarities ... : whereunto is added one book containing philosophical and prudential rules how man shall become excellent in all conditions, whether high or low, and lead his life with health of body and mind ... / written by that famous physitian, Levinus Lemnius.
   author=Lemnius, Levinus, 1505-1568.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A47/A47663.P5.xml
   tokens=130209:130249
   passage_id=d83f3ec5d4a25b366a7fc85aa34ac8da608d7fd5bc702b121a520052c628f73d
   faces , and all their bodies : for the pushes that break forth in the ourward skin grow white from melancholique burnt to ashes . the flesh indeed of this creature when it is measly is sweet and well relished
=============================================================
55. year=1630 distance=0.2711
   doc_id=A06768
   title=The Buckler of bodilie health whereby health may bee defended, and sickesse repelled: consecrate by the au[thor] the vse of his cou[...] [...]shing from his heart (though it were to his hurt) to see the fruites of his labour on the constant wellfare of all his countrie-men. By Mr. Iohn Makluire, Doctor in Medicine.
   author=Makluire, John.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A0/A06768.P5.xml
   tokens=17199:17239
   passage_id=96f15539094d4012e9e6c61faba5d2c841142131265c8087cc08baed7489fafc
   their belly still should bee keeped open . of flegmaticks . flegmaticks are of colour white or grayish , their face bowden or swelled in some kynd , the body growne , soft , cold to the touch , without
=============================================================
56. year=1629 distance=0.2712
   doc_id=A09010
   title=Paradisi in sole paradisus terrestris. or A garden of all sorts of pleasant flowers which our English ayre will permitt to be noursed vp with a kitchen garden of all manner of herbes, rootes, & fruites, for meate or sause vsed with vs, and an orchard of all sorte of fruitbearing trees and shrubbes fit for our land together with the right orderinge planting & preseruing of them and their vses & vertues collected by Iohn Parkinson apothecary of London 1629.
   author=Parkinson, John, 1567-1650.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A09/A09010.P5.xml
   tokens=40073:40113
   passage_id=ef399a71ba957e0697338e62a8a021f7d40b6262b162334b5b21cc134f121e3b
   white flambant or fooles coate . tulipa cinnabarina albo flammata . the vermillion flamed . tulipa plumata rubra lutea . the feathered tulipa red and yellow . depiction of flowers le suisse , tenijs radiata magnis ex rubore pallore .
=============================================================
57. year=1820 distance=0.2714
   doc_id=CLMET3149
   title=Prometheus unbound
   author=Shelley, Percy Bysshe
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_149.txt
   tokens=24540:24580
   passage_id=f10d725a458a444b75f1ed3791121341f16824496582ab1e7a910d85238617d2
   white light Scattered in strings ; yet its two eyes are heavens 225 Of liquid darkness , which the Deity Within seems pouring , as a storm is poured From jagged clouds , out of their arrowy lashes , Tempering
=============================================================
58. year=1650 distance=0.2715
   doc_id=B02506
   title=The compleat bell-man. Being a pattern for all sorts of people to take notice of the most remarkable times and dayes in the year. Very usefull for to read and meditate upon at all times, both in sicknesse and in health. Published for the help of all those that desire to be godly in this life, and eternally happy in the life to come. / By H: Crouch.
   author=Crouch, Humphrey, fl. 1635-1671.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/B02/B02506.P5.xml
   tokens=1186:1226
   passage_id=c14c98455d3726ec291d96a03da8e58303bd302118a7f2c8eaf4b18ff75d9e1b
   day and shortest night , the summer now wee count half spent , vvhich young folk spend in merryment , in pretty sports and harmlesse play , wishing that summer still would stay . for st . iohn baptists day
=============================================================
59. year=1640 distance=0.2715
   doc_id=A09011
   title=Theatrum botanicum: = The theater of plants. Or, An herball of a large extent containing therein a more ample and exact history and declaration of the physicall herbs and plants that are in other authours, encreased by the accesse of many hundreds of new, rare, and strange plants from all the parts of the world, with sundry gummes, and other physicall materials, than hath beene hitherto published by any before; and a most large demonstration of their natures and vertues. Shevving vvithall the many errors, differences, and oversights of sundry authors that have formerly written of them; and a certaine confidence, or most probable conjecture of the true and genuine herbes and plants. Distributed into sundry classes or tribes, for the more easie knowledge of the many herbes of one nature and property, with the chiefe notes of Dr. Lobel, Dr. Bonham, and others inserted therein. Collected by the many yeares travaile, industry, and experience in this subject, by Iohn Parkinson apothecary of London, and the Kings herbarist. And published by the Kings Majestyes especial
   author=Parkinson, John, 1567-1650.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A09/A09011.P5.xml
   tokens=181592:181632
   passage_id=70ac15840e20b95da77909aaa93e50cf76aa13cabc04bb4aee040a2d94546f41
   those of alypum : the flowers are white and small , set close together , in a long tuft , but so covered with the white woollinesse that they can scarse bee perceived : after which come small blacke seede
=============================================================
60. year=1650 distance=0.2715
   doc_id=B02506
   title=The compleat bell-man. Being a pattern for all sorts of people to take notice of the most remarkable times and dayes in the year. Very usefull for to read and meditate upon at all times, both in sicknesse and in health. Published for the help of all those that desire to be godly in this life, and eternally happy in the life to come. / By H: Crouch.
   author=Crouch, Humphrey, fl. 1635-1671.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/B02/B02506.P5.xml
   tokens=1164:1204
   passage_id=e10ce4d657133109b9454bd509e7dba5a4d3cf50823119a3b8505aa1aef66879
   and keep us ever from the spirit of error . for barnaby bright . this day wee call barnaby bright the longest day and shortest night , the summer now wee count half spent , vvhich young folk spend in
=============================================================
61. year=1661 distance=0.2719
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=948036:948076
   passage_id=2fa56ab588d9ed5aeebf0bed30dcb9239ef3b8593bb4b73ec0028679fac9b64f
   was partaker with his children of flesh and bloud , heb . . . . white denoteth his innocency in himself without spot of sin : red , his sinfull case by imputation , cor . . . and so
=============================================================
62. year=1836 distance=0.2724
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=1983:2023
   passage_id=ec1a12c0d9b16a4f8034cc1d7f1a8c3dd00c57d07527d714ff5190004ec2f949
   was of a dead white , without the least appearance of what is called complexion , though some parts of his body were in a small degree less white than others : his hair , eye brows , and beard
=============================================================
63. year=1643 distance=0.2725
   doc_id=A46823
   title=A help for the understanding of the Holy Scripture intended chiefly for the assistance and information of those that use constantly every day to reade some part of the Bible, and would gladly alwayes understand what they read if they had some man to help them : the first part : containing certain short notes of exposition upon the five books of Moses, to wit Genesis, Exodus, Leviticus, Numbers, and Deuteronomie : wherein  all such passages in the text are explained as were thought likely to be questioned by any reader of ordinary capacity ... / by Arthur Jackson ...
   author=Jackson, Arthur, 1593?-1666.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A46823.P5.xml
   tokens=186426:186466
   passage_id=5ef2071982895428f0c7f4355678c964b97a7d316c471b0c45d39e8a6058d68a
   observing the colour of the hair that grew in the skinne where the bright spot was , for if the hair being of some other colour before were turned white , then it was a leprosie ; and . by
=============================================================
64. year=1847 distance=0.2725
   doc_id=CLMET3178
   title=Jane Eyre
   author=BrontÎ, Charlotte
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_178.txt
   tokens=52506:52546
   passage_id=169cda926caff4ada7db4e4831d867eb17bfe907348f8cadb63d3a13c81631e6
   grapes and vine - leaves , beneath which glowed in rich contrast crimson couches and ottomans ; while the ornaments on the pale Parian mantelpiece were of sparkling Bohemian glass , ruby red ; and between the windows large mirrors
=============================================================
65. year=1835 distance=0.2726
   doc_id=CLMET3157
   title=King O'Neil
   author=Gore, Catherine Grace Frances
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_157.txt
   tokens=185:225
   passage_id=50bd7cf8d4ebdd4af0635a0954b1e0947d51f44c1ec317e6c66bc79558611105
   as Major de Burgh ) Captain O ' Neil ( Dressed as Major de Burgh ) Rafe ( Light drab poplin coat , embroidered waistcoat , white silk breeches , three - cornered hat trimmed with white feather ) Countess
=============================================================
66. year=1847 distance=0.2727
   doc_id=CLMET3178
   title=Jane Eyre
   author=BrontÎ, Charlotte
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_178.txt
   tokens=52515:52555
   passage_id=cf21abc4e2a8ffb9ddb87c59c153ee6a9b680c4d09df9ba1cb2126b8a00e0340
   in rich contrast crimson couches and ottomans ; while the ornaments on the pale Parian mantelpiece were of sparkling Bohemian glass , ruby red ; and between the windows large mirrors repeated the general blending of snow and fire .
=============================================================
67. year=1836 distance=0.2727
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=2003:2043
   passage_id=3ba16e7675e728b994ab779a91642a461058c8ac9ac7e6badc103fb93304fc99
   his body were in a small degree less white than others : his hair , eye brows , and beard were as white as his skin ; his eyes appeared as if they were blood shot , and he seemed
=============================================================
68. year=1836 distance=0.2728
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=3952:3992
   passage_id=6bf2355d00f430f7ae59a3ac2ddd1eb1bec4bd456d8d6a6af6604e3cc9630b66
   characteristics of the xanthous . further examples from winterbottom , marcgraf , pallas , goldsmith , burchell and others describing xanthous or intermediate white negroes with red , yellow or flaxen hair , blue grey brown eyes , and fair
=============================================================
69. year=1650 distance=0.2728
   doc_id=B02506
   title=The compleat bell-man. Being a pattern for all sorts of people to take notice of the most remarkable times and dayes in the year. Very usefull for to read and meditate upon at all times, both in sicknesse and in health. Published for the help of all those that desire to be godly in this life, and eternally happy in the life to come. / By H: Crouch.
   author=Crouch, Humphrey, fl. 1635-1671.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/B02/B02506.P5.xml
   tokens=1184:1224
   passage_id=604090b8c567e76d7e79acd3f6306929130b3a46bf0e5e811d99e835e2beedfb
   the longest day and shortest night , the summer now wee count half spent , vvhich young folk spend in merryment , in pretty sports and harmlesse play , wishing that summer still would stay . for st . iohn
=============================================================
70. year=1836 distance=0.2730
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=280:320
   passage_id=62058e4b1364ed66f86e579689f5923e1e90be765998c32a6fa5cc9f7089f72e
   is the result of careful protection from the sun . and in europe , black haired women are termed brunettes , from the hue which the skin acquires on ordinary exposure . the colour of the eye corresponds with that
=============================================================
71. year=1659 distance=0.2731
   doc_id=A45436
   title=A paraphrase and annotations upon all the books of the New Testament briefly explaining all the difficult places thereof / by H. Hammond.
   author=Hammond, Henry, 1605-1660.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A45436.P5.xml
   tokens=1275563:1275603
   passage_id=0bbbdb914e8286f2463369443547558671147e7e4415d43b9dd5029475a915fb
   enemies , as he doth in the subsequent visions . after the same manner god sitting in judgment is represented dan . . . he himself , his head , hair , and garments , white as snow or wool
=============================================================
72. year=1837 distance=0.2734
   doc_id=CLMET3152
   title=The French Revolution
   author=Carlyle, Thomas
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_152.txt
   tokens=208352:208392
   passage_id=fb4a934cb4d2d8e4027b2c4d609d5b974aa0b9faafcef5fd45073eeac77691a5
   - Antoine rising in mass ; who has many other things to see . Last and grimmest of all note old Ruhl , with his brown dusky face and long white hair ; of Alsatian Lutheran breed ; a man
=============================================================
73. year=1661 distance=0.2738
   doc_id=A66593
   title=A complete Christian dictionary wherein the significations and several acceptations of all the words mentioned in the Holy Scriptures of the Old and New Testament are fully opened, expressed, explained ... / begun by ... Mr. Thomas Wilson, minister of the word at St. Georges in Canterbury ... ; continued by Mr. John Bagwell... ; augmented and enlarged with a sixfold addition ... extracted out of the most approved authors, both ancient and modern, especially Petri Ravanelli Bibliotheca sacra.
   author=Wilson, Thomas, 1563-1622.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A6/A66593.P5.xml
   tokens=948061:948101
   passage_id=99a49e6370ad44966d8d87937ef8164d7fc7f6d5ffe276c8403214315fb4cf5e
   : red , his sinfull case by imputation , cor . . . and so these two colours are used to signifie sinfulnesse and innocency , isa . . . and red betokeneth his sufferings to the shedding of his
=============================================================
74. year=1639 distance=0.2740
   doc_id=A02060
   title=The compleat horseman and expert ferrier In two bookes. The first, shewing the best manner of breeding good horses, with their choyce, nature, riding and dyeting ... The second, directing the most exact and approved manner how to know and cure all maladies and diseases in horses ... dedicated to his most Excellent Majestie, by Thomas de Gray Esquire.
   author=De Grey, Thomas.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A02/A02060.P5.xml
   tokens=13025:13065
   passage_id=2e9e141a15107be66e0f4df4efb7fca1f60d2c938c5a8d3b424470ba46c638da
   were thus marked . all good horsemen doe attest , that the horse with much white upon his face , raw nosed , sheath , yard , tuell , hooves white , skin white , and legs hosed , waleyed
=============================================================
75. year=1652 distance=0.2742
   doc_id=A57647
   title=Arcana microcosmi, or, The hid secrets of man's body discovered in an anatomical duel between Aristotle and Galen concerning the parts thereof : as also, by a discovery of the strange and marveilous diseases, symptomes & accidents of man's body : with a refutation of Doctor Brown's Vulgar errors, the Lord Bacon's natural history, and Doctor Harvy's book, De generatione, Comenius, and others : whereto is annexed a letter from Doctor Pr. to the author, and his answer thereto, touching Doctor Harvy's book De Generatione / by A.R.
   author=Ross, Alexander, 1591-1654.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A57647.P5.xml
   tokens=62735:62775
   passage_id=ded45299d6447449c7b7e960e6ec19126e1ded8991722a9075fc6d9e2eaab38e
   makes the green corn turn white and hard with his heat , and at the same time makes the plumb black and soft ; women that blanch or whiten their linnen in the sun , know that he can an
=============================================================
76. year=1659 distance=0.2742
   doc_id=A45436
   title=A paraphrase and annotations upon all the books of the New Testament briefly explaining all the difficult places thereof / by H. Hammond.
   author=Hammond, Henry, 1605-1660.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A45436.P5.xml
   tokens=1275583:1275623
   passage_id=55093c997c9f7b606319e9b9e949cc69bd1f93a7f04b53069a26e76c2434756e
   dan . . . he himself , his head , hair , and garments , white as snow or wool , then under that , the throne like a flame of fire , and under that the wheels as burning
=============================================================
77. year=1657 distance=0.2743
   doc_id=A53713
   title=Of communion with God the Father, Sonne, and Holy Ghost, each person distinctly in love, grace, and consolation, or, The saints fellowship with the Father, Sonne, and Holy Ghost, unfolded by John Owen ...
   author=Owen, John, 1616-1683.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A53713.P5.xml
   tokens=27001:27041
   passage_id=4dabef2f2f4e8bd969a02f61d76c86010bc1130d9b981f847fee3727324514d4
   as pure wooll . and of christ in his transfiguration , when he had on him a mighty lustre of the deity , his face did shine as the sunne , and his rayment was white as the light :
=============================================================
78. year=1658 distance=0.2744
   doc_id=A42668
   title=The history of four-footed beasts and serpents describing at large their true and lively figure, their several names, conditions, kinds, virtues ... countries of their breed, their love and hatred to mankind, and the wonderful work by Edward Topsell ; whereunto is now added, The theater of insects, or, Lesser living creatures ... by T. Muffet ...
   author=Topsell, Edward, 1572-1625?
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A42668.P5.xml
   tokens=211004:211044
   passage_id=fd5bd58ad23e8ac952dc8b9e2e3e0826218f37f52afd232eb5384fa10870bea0
   been white haired conies , whose skin was black , and hair of their ears black . they are bred in lybia , in scythia , and in italy , in the top of the mountains , and so brought
=============================================================
79. year=1850 distance=0.2745
   doc_id=CLMET3208
   title=Olive
   author=Craik, Dinah Maria Mulock
   filepath=clmet/corpus/txt/plain/CLMET3_1_3_208.txt
   tokens=11849:11889
   passage_id=74fa2a76e8bf5639938eb66426bec6da9969d3917d69f110deba6761776ded57
   on a lonely moor , or appearing by a cradle - side ; supernatural , yet fraught with a nameless beauty . She was dressed with the utmost care , in white , with blue ribands ; and her lovely
=============================================================
80. year=1972 distance=0.2746
   doc_id=albinism_case_report_psaila_1972
   title=Albinism – A Case Report
   author=A.J. Psaila M.D., D.C.H. (Lond.), M.R.C.P. (U.K.), M.R.C.P. (Edin.)
   filepath=misc/albinism_case_report_psaila_1972.xml
   tokens=220:260
   passage_id=e6b27695b29c565d540454fbd023b00c49d3412d1a547593e00060bb4f93647a
   has been quite normal for her age , but her weight and height are in the th percentile range . photograph shows some of the features of this child ' s pigment abnormality . her skin is extremely white which
=============================================================
81. year=1879 distance=0.2755
   doc_id=american_cyclopedia_alibi_albinos
   title=Albi Albinos
   author=(unknown)
   filepath=misc/american_cyclopedia_alibi_albinos.xml
   tokens=609:649
   passage_id=cdef9d8299edfa1401c74943e4f48732a7683aebc83b04db43fb7ab9fb46ea4a
   , with hair of the same shade , and eyes with the iris deficient in the black or blue or hazel pigment , which in others conceals the delicate network of blood vessels , and the intense red ness they
=============================================================
82. year=1638 distance=0.2756
   doc_id=A01446
   title=The historie of life and death With observations naturall and experimentall for the prolonging of life. Written by the Right Honorable Francis Lord Verulam, Viscount S. Alban.
   author=Bacon, Francis, 1561-1626.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A0/A01446.P5.xml
   tokens=6866:6906
   passage_id=3ba5129af1c08f94f9a71af610a44fbe7bafd6ac71c80d42e81cc6b60d4eeac6
   somewhat longer liv ' d than the cocke , is a testy angry bird , and hath very white flesh . . the ring dove being an ayery bird , loving to build and sit high , is long liv
=============================================================
83. year=1654 distance=0.2758
   doc_id=A45331
   title=Comarum akosmia the loathsomnesse of long haire, or, A treatise wherein you have the question stated, many arguments against it produc'd, and the most materiall arguguments [sic] for it refell'd and answer'd : with the concurrent judgement of divines both old and new against it : with an appendix against painting, spots, naked breasts, &c. / by Thomas Hall ...
   author=Hall, Thomas, 1610-1665.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A45331.P5.xml
   tokens=3690:3730
   passage_id=de853edafcf8f9a035f8337ae89f915956a3516aa45c59cc47524841d30abdc2
   in our dayes that can do both ; by powdering their haire , they can make that white which was black , ( the powder forgets the dust ; ) other by periwigs personall defects in nature or occasionall wants
=============================================================
84. year=1660 distance=0.2759
   doc_id=A49883
   title=The world surveyed, or The famous voyages & travailes of Vincent le Blanc, or White, of Marseilles ... containing a more exact description of several parts of the world, then hath hitherto been done by any other authour : the whole work enriched with many authentick histories / originally written in French ; and faithfully rendred into English by F.B., Gent.
   author=Leblanc, Vincent, 1554-ca. 1640.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A49883.P5.xml
   tokens=94585:94625
   passage_id=32d3637771e2c6d663a982234ddb4c66cffc77c7bba558844feb030f8387ca98
   makes a very pleasant prize . the hair of these apes is soft as silk , and their genitories violet colour , or pale red . there are huge munkeyes white as snow , that make a thousand ridiculous faces
=============================================================
85. year=1629 distance=0.2759
   doc_id=A09010
   title=Paradisi in sole paradisus terrestris. or A garden of all sorts of pleasant flowers which our English ayre will permitt to be noursed vp with a kitchen garden of all manner of herbes, rootes, & fruites, for meate or sause vsed with vs, and an orchard of all sorte of fruitbearing trees and shrubbes fit for our land together with the right orderinge planting & preseruing of them and their vses & vertues collected by Iohn Parkinson apothecary of London 1629.
   author=Parkinson, John, 1567-1650.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A09/A09010.P5.xml
   tokens=40093:40133
   passage_id=9e05c2aef7c7c31c82d79b5dfed721131cf4291a8d62fd049f4961254f1ad7de
   the feathered tulipa red and yellow . depiction of flowers le suisse , tenijs radiata magnis ex rubore pallore . a swisse , paned with a faire red and pale white or strawe colour . altera dicta goliah a floris
=============================================================
86. year=1658 distance=0.2761
   doc_id=A46807
   title=Annotations upon the five books immediately following the historicall part of the Old Testament (commonly called the five doctrinall or poeticall books) to wit, the book of Iob, the Psalms, the Proverbs, Ecclesiastes, and the Song of Solomon ... / by Arthur Jackson ...
   author=Jackson, Arthur, 1593?-1666.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A46807.P5.xml
   tokens=811882:811922
   passage_id=9a2399498e6d9672486d8e8ce146e8762c35c47b85f65c449ce639a60978e7bb
   on his breast and belly , to wit , that they were vey beautifull and glorious , shining white , as the brightest polished ivory , but adorned with some trimming , or bound about with some girdle of blew
=============================================================
87. year=1653 distance=0.2763
   doc_id=A35381
   title=Pharmacopœia Londinensis, or, The London dispensatory further adorned by the studies and collections of the Fellows, now living of the said colledg ... / by Nich. Culpeper, Gent.
   author=Royal College of Physicians of London.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A3/A35381.P5.xml
   tokens=11267:11307
   passage_id=bc9c273af13cdc0ebf62c0c2ab868de22ab6a06b58487bdca7c839eb8df1c7de
   . adiantum , album , nigrum . maiden hair , white and black . they are temperate , yet drying , white maiden hair is that we usually call wall rue ; they both open obstructions , clense the breast
=============================================================
88. year=1657 distance=0.2763
   doc_id=A49255
   title=The penitent pardoned a treatise wherein is handled the duty of confession of sin and the priviledge of the pardon of sin : together with a discourse of Christs ascension into heaven and of his coming again from heaven : wherein the opinion of the Chiliasts is considered and solidly confuted / being the sum and substance of several sermons preached by that faithful servant of Christ, Mr. Christopher Love ...
   author=Love, Christopher, 1618-1651.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A49255.P5.xml
   tokens=29149:29189
   passage_id=01bb1597390b04198b7a9d08a4d52581a0cd7d50a16ffab93065027b2ff4a13b
   quick and raw flesh in the rising , they understand it to be adventurousnesse in a sinner to commit sin against a raw , and a gawled , and a rebuked conscience . another man might have scabs all over
=============================================================
89. year=1658 distance=0.2766
   doc_id=A44073
   title=A scripture-catechisme towards the confutation of sundry errours, some of them of the present times / by Thomas Hodges.
   author=Hodges, Thomas, d. 1688.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A44/A44073.P5.xml
   tokens=19852:19892
   passage_id=c43b26649c61d4b536a3c645f563e24fd431f728c2b7ef05e67213b76ec537c3
   haire white or black . q . is it probable that christ was naturally of a beautifull countenance , that the son of david was like david , yea fairer then the children of men ? a . psal .
=============================================================
90. year=1640 distance=0.2767
   doc_id=A04984
   title=An easy method to know the causes and signs of the humour most ruleth in the body and to avoid thereby things hurtful: as also to enable men to give better account of the state of their bodies, when they are diseased to the physitian, and not hazard the lives of themselves and their freinds by only urging him to prophesy (and that often falsily) the disease by onely the inspection of vrine.
   author=La Framboisière, Nicolas Abraham de, b. 16th cent.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A04/A04984.P5.xml
   tokens=2495:2535
   passage_id=25b3f8ebd895054d3f63470a71bf0f46c2cc362a1838f4cbd988d718da8aaf0b
   and troubled with much setling . phlegme voluntarily breaking forth by vomit or dejection , and a body overflowing with moisture . white flux in women . the custome and frequency of phlegmatick diseases , as swellings coming of phlegme
=============================================================
91. year=1836 distance=0.2767
   doc_id=sherwood_1838_researches_into_the_physical_history_of_mankind
   title=Researches into the Physical History of Mankind
   author=James Cowles Prichard
   filepath=misc/sherwood_1838_researches_into_the_physical_history_of_mankind.xml
   tokens=2143:2183
   passage_id=9531506898e6e392e81b499852de39c3a68bc731204fc787042019fb4d0dec90
   colour , because their hair is altogether as white as their skin ; and in general their whole exterior appearance is unnatural . they have this distinguishing peculiarity , that they cannot endure the light of the broad day .
=============================================================
92. year=1972 distance=0.2768
   doc_id=albinism_case_report_psaila_1972
   title=Albinism – A Case Report
   author=A.J. Psaila M.D., D.C.H. (Lond.), M.R.C.P. (U.K.), M.R.C.P. (Edin.)
   filepath=misc/albinism_case_report_psaila_1972.xml
   tokens=892:932
   passage_id=583dcf98c9359c9a943ae605afb1b85041e2956174ea750bc6cdd39a93fdc018
   in areas exposed to light . total albinism is inherited via an autosomal recessive gene ( witkop c . j . ) . b ) partial albinism : is more common . here the hair is yellow or light brown
=============================================================
93. year=1643 distance=0.2768
   doc_id=A46823
   title=A help for the understanding of the Holy Scripture intended chiefly for the assistance and information of those that use constantly every day to reade some part of the Bible, and would gladly alwayes understand what they read if they had some man to help them : the first part : containing certain short notes of exposition upon the five books of Moses, to wit Genesis, Exodus, Leviticus, Numbers, and Deuteronomie : wherein  all such passages in the text are explained as were thought likely to be questioned by any reader of ordinary capacity ... / by Arthur Jackson ...
   author=Jackson, Arthur, 1593?-1666.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A46823.P5.xml
   tokens=186406:186446
   passage_id=20c2feefaf0d6b5a3f7876d978fb4abd7c35de15c0d26384f7129730dd010ca5
   to judge of such white bright spots , whether they were leprosies or no , to wit , . by observing the colour of the hair that grew in the skinne where the bright spot was , for if the
=============================================================
94. year=1657 distance=0.2770
   doc_id=A49255
   title=The penitent pardoned a treatise wherein is handled the duty of confession of sin and the priviledge of the pardon of sin : together with a discourse of Christs ascension into heaven and of his coming again from heaven : wherein the opinion of the Chiliasts is considered and solidly confuted / being the sum and substance of several sermons preached by that faithful servant of Christ, Mr. Christopher Love ...
   author=Love, Christopher, 1618-1651.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A4/A49255.P5.xml
   tokens=29093:29133
   passage_id=05650a0b1d6a4c2bc1442b2d1c3e54f6d9311141a11d25598b1ccb12a113ac75
   rising : the hebrew rabbins do understand two things by this law , first , the turning of the hair white in the sore , they note to be continuance in sin , living from youth to old age in
=============================================================
95. year=1656 distance=0.2771
   doc_id=A84087
   title=Pearls of eloquence, or, The school of complements Wherein ladies, gentlewomen, and schollars, may accommodate their courtly practice with gentile ceremonies, complemental, amorous, and high expressions of speaking, or writing of letters. By VV. Elder, Gent.
   author=Elder, William, fl. 1680-1700.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A84/A84087.P5.xml
   tokens=30335:30375
   passage_id=2409795cabc8b8048ee2556f1514f8e39efbd89ccfb879d24b460c619c454b5c
   sweet of sweets is ful . on her forehead . her stately front was figured from above , majestick , fair , wel polisht , high and pale , pure white that dim s the lillies of the vale .
=============================================================
96. year=1642 distance=0.2771
   doc_id=A35545
   title=The workes of Ephesus explained in a sermon before the honovrable House of Commons at their late solemne fast, April 27th 1642 / by Ioseph Caryl ...
   author=Caryl, Joseph, 1602-1673.
   filepath=eebo_all/eebo_phase2/P5_XML_OXF_Ph2/A35/A35545.P5.xml
   tokens=12566:12606
   passage_id=d75dec346e9ecbe7eea1eee288f316d674533b883a9ea1470b9fd047a72e2416
   they appeare . the second token of leprosie , was , when the skab did spread much in the skin . though it had no white haires , v . , . though it were no deeper then the skin
=============================================================
97. year=1879 distance=0.2773
   doc_id=american_cyclopedia_alibi_albinos
   title=Albi Albinos
   author=(unknown)
   filepath=misc/american_cyclopedia_alibi_albinos.xml
   tokens=749:789
   passage_id=9e7e8ad2898fd5958ee846fc8c2709876b783cbf54e18da192be28f80190e4b2
   negroes they met with on the coast of africa . with the features of the negro and the peculiar woolly form of the hair , the color of the skin and hair was white . the eye , in stead
=============================================================
98. year=1657 distance=0.2774
   doc_id=A53713
   title=Of communion with God the Father, Sonne, and Holy Ghost, each person distinctly in love, grace, and consolation, or, The saints fellowship with the Father, Sonne, and Holy Ghost, unfolded by John Owen ...
   author=Owen, John, 1616-1683.
   filepath=eebo_all/eebo_phase1/P5_XML_OXF_2015-02/A5/A53713.P5.xml
   tokens=26966:27006
   passage_id=48f9d945999d48574077aa910c06ab0781d2dce33648cfaa8ec3c3d0b6855c0b
   in that appearance of the most high , the ancient of days dan . . v . . it is said , his garment was white as snow , and the haire of his head as pure wooll . and
=============================================================
99. year=1820 distance=0.2774
   doc_id=CLMET3149
   title=Prometheus unbound
   author=Shelley, Percy Bysshe
   filepath=clmet/corpus/txt/plain/CLMET3_1_2_149.txt
   tokens=24550:24590
   passage_id=4509c464744bb22527e8cba47e60f6aac989c868c006be52e78e53b0340b24f2
   are heavens 225 Of liquid darkness , which the Deity Within seems pouring , as a storm is poured From jagged clouds , out of their arrowy lashes , Tempering the cold and radiant air around , With fire that
=============================================================
100. year=1885 distance=0.2775
   doc_id=CLMET3244
   title=Tarantella
   author=Blind, Mathilde
   filepath=clmet/corpus/txt/plain/CLMET3_1_3_244.txt
   tokens=55452:55492
   passage_id=bab39b90e54c1d571aea4a2505cd984eb52f5790f2faf605b0646b7147ca00d4
   deadly blue flame lit up everything with a livid brightness unknown today . ` ` Walls and faded wall - paintings , limbs of decapitated page : 170 goddesses gleaming white through the grass and rioting weeds , tottering columns

"""
