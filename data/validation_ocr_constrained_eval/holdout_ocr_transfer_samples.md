# Holdout OCR Transfer Samples

Proxy-only samples from `microsoft/trocr-base-handwritten`; outputs are ordered by render variant.

## identity_only

| target | candidate | clean whole OCR | candidate whole OCR | clean char OCR | candidate char OCR |
| --- | --- | --- | --- | --- | --- |
| insurancenewsnet | іnsurancenеԝsnеt | ["insurancemen .", "insurance Newsweek .", "insurance Newsweek .", "insurancemen ."] | ["insurancemen .", "insurance Newsweek .", "insurance Newsweek .", "insurancemen ."] | ["insurancenewsnet", "insurancenewsnet", "insurancenewsnet", "insurancenewsnet"] | ["insurancenewsnet", "insurancenewsnet", "insurancenewsnet", "insurancenewsnet"] |
| remolacha | rеmᴏӏacha | ["remolache .", "remolache .", "remolache .", "remolache ."] | ["remolache .", "remolache .", "remolache .", "remolache ."] | ["remolacha", "remolacha", "remolacha", "remolacha"] | ["remolacha", "remolacha", "remolacha", "remolacha"] |
| theactivetimes | theactіvеtіmеs | ["thearchives .", "theactivetimes .", "theactive times .", "theactive times ."] | ["thearchives .", "theactivetimes .", "theactive times .", "theactive times ."] | ["theactivetimes", "theactivetimes", "theactivetimes", "theactivetimes"] | ["theactivetimes", "theactivetimes", "theactivetimes", "theactivetimes"] |
| yonghui | yᴏngһuі | ["yonghui", "younghui", "yonghui", "yonghui"] | ["yonghui", "younghui", "yonghui", "yonghui"] | ["yonghui", "yonghui", "yonghui", "yonghui"] | ["yonghui", "yonghui", "yonghui", "yonghui"] |
| foxydeal | foхydеal | ["foxy deal .", "foxy deal .", "foxy deal .", "foxy deal ."] | ["foxy deal .", "foxy deal .", "foxy deal .", "foxy deal ."] | ["foxydeal", "foxydeal", "foxydeal", "foxydeal"] | ["foxydeal", "foxydeal", "foxydeal", "foxydeal"] |
| banregio | banregіо | ["banregio", "banregio", "banregio", "banregio"] | ["banregio", "banregio", "banregio", "banregio"] | ["banregio", "banregio", "banregio", "banregio"] | ["banregio", "banregio", "banregio", "banregio"] |
| dailybasis | dаiӏybаsіs | ["daily basis .", "daily basis", "daily basis .", "daily basis ."] | ["daily basis .", "daily basis", "daily basis .", "daily basis ."] | ["dailybasis", "dailybasis", "dailybasis", "dailybasis"] | ["dailybasis", "dailybasis", "dailybasis", "dailybasis"] |
| efesco | efеscο | ["efesco", "efesco", "efesco", "efesco"] | ["efesco", "efesco", "efesco", "efesco"] | ["efesco", "efesco", "efesco", "efesco"] | ["efesco", "efesco", "efesco", "efesco"] |
| sitesi | sіtеsі | ["sitesi", "sitesi", "sitesi", "sitesi"] | ["sitesi", "sitesi", "sitesi", "sitesi"] | ["sitesi", "sitesi", "sitesi", "sitesi"] | ["sitesi", "sitesi", "sitesi", "sitesi"] |
| groupspaces | grοuрspаces | ["groupspaces .", "groupspaces .", "groupspaces .", "groupspaces ."] | ["groupspaces .", "group spaces .", "group spaces .", "group spaces ."] | ["groupspaces", "groupspaces", "groupspaces", "groupspaces"] | ["groupspaces", "groupspaces", "groupspaces", "groupspaces"] |
| christ | сһrіst | ["christ .", "christ .", "christ .", "christ ."] | ["christ .", "christ .", "christ .", "christ ."] | ["christ", "christ", "christ", "christ"] | ["christ", "christ", "christ", "christ"] |
| fivepirates | fіᴠepіratеs | ["five pirates .", "five pirates .", "five pirates .", "fivepiretes ."] | ["fivepirates .", "fivepirates .", "five pirates .", "fivepiretes ."] | ["fivepirates", "fivepirates", "fivepirates", "fivepirates"] | ["fivepirates", "fivepirates", "fivepirates", "fivepirates"] |

## identity_plus_confusable_legit

| target | candidate | clean whole OCR | candidate whole OCR | clean char OCR | candidate char OCR |
| --- | --- | --- | --- | --- | --- |
| efavormart | eƭavοrmаrt | ["efavourment", "efavourment", "efavourment", "efavourment"] | ["efavourment", "efavourment", "efavourment", "efavourment"] | ["efavormart", "efavormart", "efavormart", "efavormart"] | ["efavormart", "etavormart", "eaavormart", "etavormart"] |
| frasicelebri | ƭɾаsіcelebrі | ["presicle debt .", "presiclelebril", "fresidelebril", "presiclelebril"] | ["transicelebril", "transiclelebril", "easicelebri", "transicle alert ."] | ["frasicelebri", "frasicelebri", "frasicelebri", "frasicelebri"] | ["frasicelebri", "thasicelebri", "afasicelebri", "tnasicelebri"] |
| convertico | cᴏnveɾtіco | ["converico", "converico", "convertico", "conventico"] | ["converico", "converico", "convertico", "converico"] | ["convertico", "convertico", "convertico", "convertico"] | ["convertico", "convehtico", "conveftico", "conventico"] |
| filmsforaction | filrnѕfᴏraction | ["filmsification .", "filmstoration .", "film-storation .", "filmstoration ."] | ["filmsification .", "filmsification .", "film-storation .", "filmsification ."] | ["filmsforaction", "filmsforaction", "filmsforaction", "filmsforaction"] | ["filrnsforaction", "filrnsforaction", "filrnsforaction", "filrnsforaction"] |
| proflowers | proƭlοwеrѕ | ["proflowers .", "proflowers", "proflowers", "proflowers"] | ["proflowers .", "procious", "prociousers", "prolowers ."] | ["proflowers", "proflowers", "proflowers", "proflowers"] | ["proflowers", "protlowers", "proalowers", "protlowers"] |
| fivebelow | ƭіᴠebelow | ["free below .", "five below .", "free below .", "free below ."] | ["type below .", "tive below .", "evebelow .", "type below ."] | ["fivebelow", "fivebelow", "fivebelow", "fivebelow"] | ["fivebelow", "tivebelow", "aivebelow", "tivebelow"] |
| profitf | рrоfіtƭ | ["profit .", "profit .", "profit .", "profit ."] | ["profile .", "profile .", "profile .", "profile ."] | ["profitf", "profitf", "profitf", "profitf"] | ["profitf", "profitt", "profita", "profitt"] |
| tamindir | tаrnіndіr | ["terminer .", "terminer .", "tamindir .", "terminer ."] | ["terminer .", "terminer .", "terminable", "terminer ."] | ["tamindir", "tamindir", "tamindir", "tamindir"] | ["tarnindir", "tarnindir", "tarnindir", "tarnindir"] |
| letsforum | lеtѕƭоɾum | ["letsforum .", "letsforum .", "letsforum .", "letsforum ."] | ["leistorum .", "leistorum .", "letsforum .", "leistorum ."] | ["letsforum", "letsforum", "letsforum", "letsforum"] | ["letsforum", "letstohum", "letsaofum", "letstonum"] |
| statesmanjournal | statesrnanjᴏuɾnaƚ | ["statesmanjournal", "statesmanjournal", "statesmanjournal", "statesmanjournal"] | ["statesmanjournat", "statesmanjournat", "statesmanjournat", "statesmanjournat"] | ["statesmanjournal", "statesmanjournal", "statesmanjournal", "statesmanjournal"] | ["statesrnanjourna1", "statesrnanjouhnar", "statesrnanjoufna1", "statesrnanjounnat"] |
| doterra | dоtеɾɾа | ["doters .", "dolera", "dotierra", "dolerre"] | ["doterna", "dotensa", "dotetta", "doterna"] | ["doterra", "doterra", "doterra", "doterra"] | ["doterra", "dotehha", "doteffa", "dotenna"] |
| alltrails | аƚӏtrаils | ["all credits .", "allinois .", "allitrafts .", "allinois ."] | ["effurells .", "effurets .", "effuratis", "efforts ."] | ["alltrails", "alltrails", "alltrails", "alltrails"] | ["a1ltrails", "arltrails", "a1ltrails", "atltrails"] |
