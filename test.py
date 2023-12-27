check = { "a":4, "c":3, "b":12 }    
check_view = [ (v,k) for k,v in check.items() ]
check_view.sort() # natively sort tuples by first element
for v,k in check_view:
    print(f"{k}: {v}")