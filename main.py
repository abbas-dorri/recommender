from CollabItemRecommender import CollabItemRecommender
from ContentRecommender import ContentRecommender
from HybridRecommender import HybridRecommender

def askInput(cbi : CollabItemRecommender, cr : ContentRecommender, hr : HybridRecommender):
    print('enter option: ')
    try:
        ans = int(input())
    except:
        askInput(cbi,cr,hr)

    print('enter title: ')
    title = input()

    if(ans == 1):
        cbi.recommend_SD(title)
    if(ans == 2):
        cbi.train_and_save_SD()
    if(ans == 3):
        cbi.load_recommend_SD(targetTitle=title)
    if(ans == 4):
        cbi.recommend_LD(title)
    if(ans == 5):
        cbi.train_and_save_LD()
    if(ans == 6):
        cbi.load_recommend_LD(targetTitle=title)
    
    if(ans == 7):
        cr.recommend_SD(title)
    if(ans == 8):
        cr.train_and_save_SD()
    if(ans == 9):
        cr.load_recommend_SD(targetTitle=title)
    if(ans == 10):
        cr.recommend_LD(title)
    if(ans == 11):
        cr.train_and_save_LD()
    if(ans == 12):
        cr.load_recommend_LD(targetTitle=title)

    if(ans == 13):
        hr.recommend_SD(title)
    if(ans == 14):
        hr.train_and_save_SD()
    if(ans == 15):
        hr.load_recommend_SD(targetTitle=title)

    if(ans == 69):
        cbi.load_recommend_SD_TEST()
    if(ans == 70):
        cr.load_recommend_SD_TEST()

    if(ans == 99):
        exit()

    askInput(cbi,cr,hr)

if __name__ == '__main__':
    print()
    print("whats your mode: first ones are collab")
    print("                 1 = train and recommend | 4 = ' ' for larger dataset")
    print("                 2 = train and save      | 5 = ' ' for larger dataset")
    print("                 3 = load and recommend  | 6 = ' ' for larger dataset")
    print()    
    print("from here on its content:               ")
    print("                 7 = train and recommend | 10 = ' ' for larger dataset")
    print("                 8 = train and save      | 11 = ' ' for larger dataset")
    print("                 9 = load and recommend  | 12 = ' ' for larger dataset")
    print()    
    print("from here on its hybrid content collab:               ")
    print("                 13 = train and recommend")
    print("                 14 = train and save     ")
    print("                 15 = load and recommend ")
    print()
    print("                 69 = TEST collab-item load and recommend ")
    print("                 70 = TEST content load and recommend ")
    
    cr = ContentRecommender()
    cbi = CollabItemRecommender()
    hr = HybridRecommender()
    askInput(cbi,cr,hr)