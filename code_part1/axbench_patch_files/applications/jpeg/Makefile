6,7c6,7
< CFLAGS		:= -Wall -lfann -lboost_regex -std=c++11
< LFLAGS		:= -lfann -lboost_regex -lParrot
---
> CFLAGS		:= -Wall -std=c++11
> LFLAGS		:= -lParrot
35,36c35,36
< 	python $(PLANG) -c $(PARROT_JSON) -a observe -e "g++ $(CFLAGS) $(INCLUDE) -I$(PARROT_LIB) -c" -s "$<" -o "$@" -t
< 	#$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@
---
> 	#python $(PLANG) -c $(PARROT_JSON) -a observe -e "g++ $(CFLAGS) $(INCLUDE) -I$(PARROT_LIB) -c" -s "$<" -o "$@" -t
> 	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@
