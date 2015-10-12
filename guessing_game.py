
# coding: utf-8

# In[1]:

import sys
import random 


# In[43]:

def play():
    Enter = "Pls press enter to continue ..... "
    print "Welcome to the Guessing game " 
    print "Pls guess a number between 1-10 "

    computer_num=random.randint(1,10)
    guess1= raw_input("My first guess is : ")
    guess1=int(guess1)
    if (guess1 == computer_num) :
        print "You win"
        raw_input(  Enter  )
        sys.exit(0)
    elif (guess1<computer_num):
            print "Sorry try again "
            print "You have 2 guesses remaining "
            print "Try to increase your guess bro !! "
            guess2= raw_input("My second guess is : ")
            guess2=int(guess2)
            if (guess2 == computer_num) :
                print "You win"
                raw_input(  Enter  )
                sys.exit(0)
            elif (guess2<computer_num):
                print "Sorry try again "
                print "You have 1 guess remaining "
                print "Try to increase your guess more more  bro !! "
                guess3= raw_input("My Third guess is : ")
                guess3=int(guess3)
                if (guess3 == computer_num) :
                    print "You win"
                    raw_input(  Enter  )
                    sys.exit(0)
                else: 
                    decision= raw_input("Sorry!! you finished your chances !! Want to play the game again ??(y/n)  ")
                    decision=str(decision)
                    decision=decision.upper()
                    if(decision==Y):
                        play()
                    else:
                        print "nice playing with you "
                        sys.exit(0)
            else:
                print "Sorry try again "
                print "You have 1 guess remaining "
                print "Try to decrease your now guess  bro !! "
                guess3= raw_input("My Third guess is : ")
                guess3=int(guess3)
                if (guess3 == computer_num) :
                    print "You win"
                    raw_input(  Enter  )
                    sys.exit(0)
                else: 
                    decision= raw_input("Sorry!! you finished your chances !! Want to play the game again ??(y/n)  ")
                    decision=str(decision)
                    decision=decision.upper()
                    if(decision==Y):
                        play()
                    else:
                        print "nice playing with you "
                        sys.exit(0)
    else:
        print "Sorry try again "
        print "You have 2 guesses remaining "
        print "Try to decrease your guess bro !! "
        guess2= raw_input("My second guess is : ")
        guess2=int(guess2)
        if (guess2 == computer_num) :
                print "You win"
                raw_input(Enter)
                sys.exit(0)
        elif (guess2<computer_num):
                print "Sorry try again "
                print "You have 1 guess remaining "
                print "Try to increase your guess now bro !! "
                guess3= raw_input("My Third guess is : ")
                guess3=int(guess3)
                if (guess3 == computer_num) :
                    print "You win"
                    raw_input(  Enter  )
                    sys.exit(0)
                else: 
                    decision= raw_input("Sorry!! you finished your chances !! Want to play the game again ?? (y/n) ")
                    decision=str(decision)
                    decision=decision.upper()
                    if(decision==Y):
                        play()
                    else:
                        print "nice playing with you "
                        sys.exit(0)
        else:
            print "Sorry try again "
            print "You have 1 guess remaining "
            print "Try to decrease your guess more more bro !! "
            guess3= raw_input("My Third guess is : ")
            guess3=int(guess3)
            if (guess3 == computer_num) :
                print "You win"
                raw_input(  Enter  )
                sys.exit(0)
            else: 
                decision= raw_input("Sorry!! you finished your chances !! Want to play the game again ??(y/n) ")
                decision=str(decision)
                decision=decision.upper()
                if(decision=='Y'):
                    play()
                else:
                    print "nice playing with you "
                    sys.exit(0)


# In[44]:

play()


# In[ ]:



