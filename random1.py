from scipy.io import loadmat

data = loadmat(r"C:\Users\anish\OneDrive\Desktop\SRIP\gaze\predicting-human-gaze-beyond-pixels\data\eye\fixations.mat")

# print(data.keys())

fix = data["fixations"]

# print(fix.shape)
# print(fix.dtype)
# print(type(fix))
# print(fix.shape)
# print(fix.dtype)

# print(type(fix[0,0]))

# img = fix[0,0]

# print(type(img))
# print(img.shape)
# print(img.dtype)
# print(img.dtype.names)
# print(img)

# img = fix[0,0]

# print(img['img'])
# print(type(img['subjects']))
# print(img['subjects'].shape)

# subs = img['subjects'][0,0]

# print(type(subs))
# print(subs.shape)
# print(subs.dtype)

# img = fix[0,0]

# subs = img['subjects'][0,0]

# print(type(subs[0,0]))
# print(subs[0,0].dtype)
# print(subs[0,0].shape)

img = fix[0,0]
subs = img['subjects'][0,0]
s = subs[0,0]

print(s['fix_x'][0,0].shape)
print(s['fix_y'][0,0].shape)
print(s['fix_duration'][0,0].shape)

print(s['fix_x'][0,0][:10])
print(s['fix_y'][0,0][:10])
print(s['fix_duration'][0,0][:10])