import pandas as pd

# Load the data from the CSV file
data = pd.read_csv("train_submission.csv")


# Function to categorize languages based on their alphabet
def categorize_language(text):
    alphabets = {
        "latin_alphabet": set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "cyrillic_alphabet": set(
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        ),
        "greek_alphabet": set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"),
        "amharic_alphabet": set(
            "ሀሁሂሃሄህሆሇለሉሊላሌልሎሏመሙሚማሜምሞሟሠሡሢሣሤሥሦሧረሩሪራሬርሮሯሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿቀቁቂቃቄቅቆቇቐቑቒቓቔቕቖቘበቡቢባቤብቦቧቨቩቪቫቬቭቮቯተቱቲታቴትቶቷቸቹቺቻቼችቾቿኀኁኂኃኄኅኆኇነኑኒናኔንኖኗኘኙኚኛኜኝኞኟአኡኢኣኤእኦኧከኩኪካኬክኮኯኰኲኳኴኵኸኹኺኻኼኽኾ኿ዀ዁ዂዃዄዅ዆዇ወዉዊዋዌውዎዏዐዑዒዓዔዕዖ዗ዘዙዚዛዜዝዞዟዠዡዢዣዤዥዦዧየዩዪያዬይዮዯደዱዲዳዴድዶዷዸዹዺዻዼዽዾዿጀጁጂጃጄጅጆጇገጉጊጋጌግጎጏጐ጑ጒጓጔጕ጖጗ጘጙጚጛጜጝጞጟጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷጸጹጺጻጼጽጾጿፀፁፂፃፄፅፆፇፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖፗፘፙፚ፛፜፝፞፟፠፡።፣፤፥፦፧፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼"
        ),
        "santali_alphabet": set("ᱚᱛᱜᱝᱞᱟᱠᱡᱢᱣᱤᱥᱦᱧᱨᱩᱪᱫᱬᱭᱮᱯᱰᱱᱲᱳᱴᱵᱶᱷᱸᱹᱺᱻᱼᱽ᱾᱿"),
        "arabic_alphabet": set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئىىة،"),
        "devanagari_alphabet": set(
            "ऀँंःअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहऽािीुूृॄॅॆेैॉोौ्ॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ।॥०१२३४५६७८९॰"
        ),
        "lao_alphabet": set(
            "ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮຯະັາິີຶືຸູົຼຽເແໂໃໄໆໜໝໞໟ໠໡໢໣໤໥໦໧໨໩໪໫໬໭໮໯໰໱໲໳໴໵໶໷໸໹໺໻໼໽໾໿"
        ),
        "oriya_alphabet": set(
            "ଅଆଇଈଉଊଋଌଏଐଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହଽାିୀୁୂୃୄେୈୋୌ୍ୖୗଡ଼ଢ଼ୟୠୡୢୣ୦୧୨୩୪୫୬୭୮୯"
        ),
        "hebrew_alphabet": set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ"),
        "georgian_alphabet": set("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"),
        "inuktitut_alphabet": set(
            "ᐃᐄᐅᐆᐊᐋᐸᐹᑕᑖᑭᑮᒃᒄᒥᒦᓄᓅᓯᓰᓱᓲᓴᓵᔭᔮᕐᕑᕕᕖᕿᖀᖁᖃᖄᖅᖆᖏᖐᖑᖓᖔᖕᖖᖠᖡᖢᖤᖥᖦᖨᖩᖪᖫᖬᖭᖮᖯᖰᖱᖲᖳᖴᖵᖶᖷᖸᖹᖺᖻᖼᖽᖾᖿᗀᗁᗂᗃᗄᗅᗆᗇᗈᗉᗊᗋᗌᗍᗎᗏᗐᗑᗒᗓᗔᗕᗖᗗᗘᗙᗚᗛᗜᗝᗞᗟᗠᗡᗢᗣᗤᗥᗦᗧᗨᗩᗪᗫᗬᗭᗮᗯᗰᗱᗲᗳᗴᗵᗶᗷᗸᗹᗺᗻᗼᗽᗾᗿ"
        ),
        "gujarati_alphabet": set(
            "અઆઇઈઉઊઋઌએઐઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહઽાિીુૂૃૄૅ૆ેૈૉોૌ્ૐૠૡૢૣ૦૧૨૩૪૫૬૭૮૯"
        ),
        "thai_alphabet": set(
            "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฯะัาิีึืฺุู฿เแโใไๅๆ็่้๊๋์ํ๎๏๐๑๒๓๔๕๖๗๘๙๚๛"
        ),
        "kannada_alphabet": set(
            "ಅಆಇಈಉಊಋಌಎಏಐಒಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಽಾಿೀುೂೃೄೆೇೈೊೋೌ್ೕೖೞೠೡೢೣ೦೧೨೩೪೫೬೭೮೯"
        ),
        "malayalam_alphabet": set(
            "അആഇഈഉഊഋഌഎഏഐഒഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറാിീുൂൃൄെേൈൊോൌ്ൗൎ൏൐൑൒൓ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൤൥൦൧൨൩൪൫൬൭൮൯"
        ),
        "tibetan_alphabet": set(
            "ཀཁགངཅཆཇཉཏཐདནཔཕབམཙཚཛཝཞཟའཡརལཤསཧཨ྄ཱིེོྀྀུྂྃ྅྆྇ྈྉྊྋྌྍྎྏྐྑྒྒྷྔྕྖྗྙྚྛྜྜྷྞྟྠྡྡྷྣྤྥྦྦྷྨྩྪྫྫྷྭྮྯྰྱྲླྴྵྶྷྸྐྵྺྻྼ྽྾྿࿀࿁࿂࿃࿄࿅࿆࿇࿈࿉࿊࿋࿌࿎࿏࿐࿑࿒࿓࿔࿕࿖࿗࿘࿙࿚࿛࿜࿝࿞࿟࿠࿡࿢࿣࿤࿥࿦࿧࿨࿩࿪࿫࿬࿭࿮࿯࿰࿱࿲࿳࿴࿵࿶࿷࿸࿹࿺࿻࿼࿽࿾࿿"
        ),
        "chinese_alphabet": set(
            "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从知全才两长儿意正实四五力理她本头高夫回位因由老更美什最书水话儿女体者但方后行者"
        ),
        "armenian_alphabet": set("ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖև"),
        "tamil_alphabet": set("அஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநபமயரலவழளறனஷஸஹக்ஷ"),
        "japanese_alphabet": set(
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
        ),
        "maldivian_alphabet": set("ހށނރބޅކއވމފދތލގޏސޑޒޓޔޕޖޗޘޙޚޛޜޝޞޟޠޡޢޣޤޥަާިީުޫެޭޮޯް"),
        "urdu_alphabet": set("اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگگلمنںنوہءھیے"),
        "sinhala_alphabet": set(
            "අආඇඈඉඊඋඌඍඎඏඐඑඒඓඔඕඖකඛගඝඞචඡජඣඤටඨඩඪණතථදධනපඵබභමයරලවශෂසහළෆාැෑිීුූෘෙේෛොෝෞ්ෟාැෑිීුූෘෙේෛොෝෞ්ෟ"
        ),
        "punjabi_alphabet": set("ਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸ਼ਸਹਾਿੀੁੂੇੈੋੌ੍ੰੱੲੳੴੵ"),
        "burmese_alphabet": set(
            "ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဣဤဥဦဧဩဪါာိီုူေဲဳဴဵံ့း္်ျြွှဿ၀၁၂၃၄၅၆၇၈၉"
        ),
        "korean_alphabet": set(
            "가각간갇갈감갑값갓갔강갖갗같개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜걤걥걧걩걬걱걲걷걸검겁것겄겅겆겉게겐겔겜겝겟겠겡겨격겪견결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆"
        ),
        "khmer_alphabet": set(
            "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑៓។៕៖ៗ៘៙៚៛ៜ៝៞៟០១២៣៤៥៦៧៨៩៪៫៬៭៮៯៰៱៲៳៴៵៶៷៸៹៺៻៼៽៾៿"
        ),
        "bengali_alphabet": set(
            "অআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৠৡ০১২৩৪৫৬৭৮৯"
        ),
    }

    # Get the set of characters in the text
    text_chars = set(text)

    # Check if the text contains characters from any of the alphabets
    for alphabet, alphabet_chars in alphabets.items():
        if text_chars & alphabet_chars:
            return alphabet

    # If the text does not match any of the alphabets, return "other"
    return "other"


# Apply the function to the Text column and create a new column for the language category
data["Language_Category"] = data["Text"].apply(categorize_language)

# Display the first few rows of the dataframe
print(data.head())

# Display the value counts for the Language_Category column
print(data["Language_Category"].value_counts())

# Display 10 sentences from other language category
print(data[data["Language_Category"] == "other"][["Text", "Label"]].head(10))
