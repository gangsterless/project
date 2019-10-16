#人类最大浏览量
import time
MAX_HUMAN_VISIT_TIMES = 2000
BEHAVIOR_MAP = {0:"visit",1:"colletion",2:"cart",3:"buy"}
PREDICTDAY = time.strptime('2014-12-18 00', "%Y-%m-%d %H")
TRAINSTARTTIME = time.strptime("2014-12-14 00", "%Y-%m-%d %H")
TRAINENDTIME  = time.strptime("2014-12-16 23", "%Y-%m-%d %H")