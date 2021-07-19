7,8c7,8
< CFLAGS		:= -Wall -lfann -lboost_regex -std=c++11
< LFLAGS		:= -lfann -lboost_regex -lParrot
---
> CFLAGS		:= -Wall -std=c++11
> LFLAGS		:= -lParrot
37,38c37,38
< 	python $(PLANG) -c $(PARROT_JSON) -a observe -e "g++ $(CFLAGS) $(INCLUDE) -I$(PARROT_LIB) -c" -s "$<" -o "$@" -t
< 	#$(CC) -D$(NPU_MODE) $(CFLAGS) $(INCLUDE) -c $< -o $@
---
> 	#python $(PLANG) -c $(PARROT_JSON) -a observe -e "g++ $(CFLAGS) $(INCLUDE) -I$(PARROT_LIB) -c" -s "$<" -o "$@" -t
> 	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@
