import cv2
import numpy as np

class OBJ:
    def __init__(self, filename, swapyz=False, texture_file=None):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        
        for line in open(filename, "r"):
        #for line in open(filename, 'r', encoding='UTF-8'):
        
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                #self.faces.append((face, norms, texcoords))
                self.faces.append([face, norms, texcoords])
                
        if texture_file is not None:
            self.texture = cv2.imread(texture_file)
            for f in self.faces:
                print("f is:\n",f)
                f.append(self.decide_face_color(f[-1], self.texture, self.texcoords))
    def decide_face_color(self, hex_color, texture, textures):
        #doesnt use proper texture
        #takes the color at the mean of the texture coords

        h, w, _ = texture.shape
        col = np.zeros(3)
        coord = np.zeros(2)
        all_us = []
        all_vs = []

        for i in hex_color:
            print("i:",i)
            t = textures[i - 1]
            t = list(textures[i-1])
            print(t)
            if t != []:
                coord = np.array([t[0], t[1]])
                u , v = int(w*(t[0]) - 0.0001), int(h*(1-t[1])- 0.0001)
                all_us.append(u)
                all_vs.append(v)
            print("all_us",all_us)
            print("all_vs",all_vs)

        u = int(sum(all_us)/len(all_us))
        v = int(sum(all_vs)/len(all_vs))

        # all_us.append(all_us[0])
        # all_vs.append(all_vs[0])
        # for i in range(len(all_us) - 1):
        #     texture = cv2.line(texture, (all_us[i], all_vs[i]), (all_us[i + 1], all_vs[i + 1]), (0,0,255), 2)
        #     pass    

        col = np.uint8(texture[v, u])
        col = [int(a) for a in col]
        col = tuple(col)
        return (col)