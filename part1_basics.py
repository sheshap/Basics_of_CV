import cv2
import numpy as np
import math


class Blend:
    def __init__(self):
        self.gaussian_pyramid = []
        self.laplacian_pyramid = []
        self.n = 5
        self.first_1D = np.array([1/4,2/4,1/4])
        self.second_1D = np.transpose([self.first_1D])
        self.drag_start = None
        self.sel = (0, 0, 0, 0)
        self.done = None

    def getSelection(self,img):
        def onmouse(event, x, y, flags, param):
            global drag_start, sel
            if event == cv2.EVENT_LBUTTONDOWN:
                drag_start = x, y
                sel = 0, 0, 0, 0
            elif event == cv2.EVENT_LBUTTONUP:
                drag_start = None
            elif drag_start:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    minpos = min(drag_start[0], x), min(drag_start[1], y)
                    maxpos = max(drag_start[0], x), max(drag_start[1], y)
                    sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
                    tmp = img.copy()
                    cv2.rectangle(tmp, (sel[0], sel[1]), (sel[2], sel[3]), (0, 255, 0), 1)
                    cv2.imshow("select", tmp)
                    #sel = sel[0],0,sel[2],img.shape[1]
        cv2.namedWindow("select", 1)
        cv2.setMouseCallback("select", onmouse)
        cv2.imshow("select", img)
        if (cv2.waitKey() & 255) == 27:
            cv2.destroyAllWindows()
            #print(sel)
            return sel

    def selectResize(self,img, width=800):
        roi1 = np.array(self.getSelection(img))
        patch = img[int(roi1[1]):int(roi1[3]), int(roi1[0]):int(roi1[2])]
        return patch

    def convolve(self, I, H):
        kernel = np.flipud(np.fliplr(H))
        ksize = kernel.shape[0]
        output = np.zeros_like(I)
        t = kernel.shape[0]-1
        imagepad = np.zeros(((I.shape[0] + 2 * t), (I.shape[1] + 2 * t), 3))
        imagepad = I
        for x in range(imagepad.shape[0]-kernel.shape[0]+1):
            for y in range(imagepad.shape[1]-kernel.shape[1]+1):
                for z in range(3):
                    output[x, y, z] = (kernel * imagepad[x:x + kernel.shape[0],y:y + kernel.shape[1], z]).sum()
        return output

    def reduce(self, I):
        g1Df1 = self.first_1D # gaussian 1D row filter
        g1Df2 = self.second_1D #separable 1D kernel in transpose
        ksize = g1Df1.shape[0]  #size of separable kernel
        output = np.zeros_like(I)   #
        t = g1Df1.shape[0]-1
        imagepad = np.zeros(((I.shape[0]+2*t), (I.shape[1]+2*t), 3)) #numpy 2D array to hold both image and padding
        imagepad[t:-t, t:-t, ] = I #
        imagepad = I
        #Gaussian blurring
        for x in range(imagepad.shape[0]-ksize+1):
            for y in range(imagepad.shape[1]-ksize+1):
                for z in range(3):
                    output[x, y, z] = (g1Df1 * (g1Df2 * imagepad[x:x + ksize, y:y + ksize, z])).sum()

        out = output
        x = (out.shape[0]+1) // 2
        y = (out.shape[1]+1) // 2
        reduced_image = cv2.resize(out,dsize=(y,x))
        #if I.shape[0] % 2 == 0  and I.shape[1]%2 ==0:
        #    return out[::2,::2,]
        #else:
        return reduced_image

    def expand(self, I, d_size):
        y,x = d_size
        newI = np.zeros((x, y, 3))
        return cv2.resize(I,dsize=d_size)

    def GaussianPyramid(self, I, n):
        iimg = I
        # Given image is at the base of the gaussian pyramid
        self.gaussian_pyramid.append(iimg)
        for i in range(1, n):
            #reduce the image
            iimg = self.reduce(iimg)
            #append/place the reduced in next level up of the gaussian pyramid
            self.gaussian_pyramid.append(iimg)
        return self.gaussian_pyramid

    def LaplacianPyramid(self, I, n):
        # top level image in gaussian pyramid is the laplacian image on top of the laplacian pyramid
        # laplacian pyramid is built upside down
        i_1_img = self.gaussian_pyramid[n - 1]
        # append the image to the laplacian pyramid list of images
        self.laplacian_pyramid.append(i_1_img)
        for i in range(2, n + 1):
            # extract the next level below in the gaussian pyramid
            i_img = self.gaussian_pyramid[n - i]
            # pass it's size to match the corresponding level laplacian image to expand to
            d_size = (i_img.shape[1],i_img.shape[0])
            # subtract the expanded image with corresponding gaussian image to get the laplacian image
            limg = cv2.subtract(i_img, self.expand(i_1_img,d_size))
            i_1_img = i_img
            #pass this laplacian image to next level for further building of pyramid
            self.laplacian_pyramid.append(limg)
        return self.laplacian_pyramid

    def reconstruct(self, n):
        reconstruct_py = []
        error_py = []
        #laplacian at the top of the laplacian pyramid is the same as gaussian and also the reconstructed image
        lap_py_i = self.laplacian_pyramid[0]
        for i in range(2, n + 1):
            gaussian = self.gaussian_pyramid[n - i]
            # g = cv2.imread("gp"+str(i)+".png")
            laplacian = self.laplacian_pyramid[i - 1]
            # l = cv2.imread("lp" + str(i) + ".png")
            d_size = (gaussian.shape[1],gaussian.shape[0])
            # expand and add to the laplacian at next level to reconstruct at next level
            recon_i = self.expand(lap_py_i,d_size) + laplacian
            #difference or error due to reconstruction
            error_i = recon_i - gaussian
            error_py.append(error_i)
            cv2.imwrite("error"+str(i)+".png",error_i)
            reconstruct_py.append(recon_i)
            cv2.imwrite("recon"+str(i)+".png", recon_i)
            lap_py_i = recon_i
        return reconstruct_py, error_py

def main():
    option_code = input("Press Option 1:Convolve, 2:Reduce, 3:Expand, 4:GaussianPyramid, 5:LaplacianPyramid, 6:ReconstructionPyramid :: ")
    if option_code == "1":
        convolve_picture = input("Please enter the picture name you want to convolve (e.g. Picture1.png): ")
        convolve_kernel = input(" Enter an nxn kernel (e.g [[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]):")
        print("Press escape on the image windows to exit")

        C = Blend()
        CI = cv2.imread(convolve_picture)
        H = np.array(eval(convolve_kernel))
        cv2.imwrite("Convolved_Image.png",C.convolve(CI,H))
        to_show_img = cv2.imread("Convolved_Image.png",1)
        cv2.imshow("Original Image",CI)
        cv2.imshow("Convolved Image",to_show_img)

    elif option_code == "2":
        reduce_image = input("Enter picture name to reduce size: ")
        reduce_kernel_1 = input("Enter first separable 1-D kernel (e.g. [1/4,2/4,1/4]): ")
        R = Blend()
        R.first_1D = np.array(eval(reduce_kernel_1))
        reduce_kernel_2 = input("Enter separable second 1-D kernel (e.g. [1/4,2/4,1/4]): ")
        R.second_1D = np.transpose([np.array(eval(reduce_kernel_2))])
        RI = cv2.imread(reduce_image, 1)
        m_reduce = np.zeros(RI.shape)
        m_norm = cv2.normalize(RI, m_reduce, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("Reduced_Image.png", R.reduce(m_norm))
        to_show_img = cv2.imread("Reduced_Image.png", 1)
        cv2.imshow("Original Image", RI)
        cv2.imshow("Reduced Image", to_show_img)

    elif option_code == "3":
        expand_image = input("Enter picture name to expand size: ")
        E = Blend()
        EI = cv2.imread(expand_image, 1)
        m_expand = np.zeros(EI.shape)
        m_norm = cv2.normalize(EI, m_expand, 0, 255, cv2.NORM_MINMAX)
        d_size = (2*EI.shape[1],2*EI.shape[0])
        cv2.imwrite("Expanded_Image.png", E.expand(m_norm,d_size))
        to_show_img = cv2.imread("Expanded_Image.png", 1)
        cv2.imshow("Original Image", EI)
        cv2.imshow("Expanded Image", to_show_img)

    elif option_code == "4":
        gaussian_image_1 = input("Enter picture name to build gaussian pyramid: ")
        num_levels = input("Enter number of levels in the pyramid (n): ")
        reduce_kernel_1 = input("Enter separable 1-D kernel (e.g. [1/4,2/4,1/4]) : ")
        G = Blend()
        G.first_1D = np.transpose(np.array(eval(reduce_kernel_1)))
        GI = cv2.imread(gaussian_image_1, 1)
        g_image = np.zeros(GI.shape)
        g_norm = cv2.normalize(GI, g_image, 0, 255, cv2.NORM_MINMAX)
        gp = G.GaussianPyramid(g_norm,int(num_levels))
        for i in range(int(num_levels)):
            cv2.imshow("Gaussian Image at level "+str(i+1), gp[i])
            cv2.imwrite("Gaussian Image at level " + str(i + 1)+".png", gp[i])

    elif option_code == "5":
        gaussian_image_1 = input("Enter picture name to build laplacian pyramid: ")
        num_levels = input("Enter number of levels in the pyramid (n): ")
        print("Need gaussian pyramid to build laplacian pyramid")
        reduce_kernel_1 = input("Enter separable 1-D kernel (e.g. [1/4,2/4,1/4]) : ")
        L = Blend()
        L.first_1D = np.transpose(np.array(eval(reduce_kernel_1)))
        GI = cv2.imread(gaussian_image_1, 1)
        g_image = np.zeros(GI.shape)
        g_norm = cv2.normalize(GI, g_image, 0, 255, cv2.NORM_MINMAX)
        gp = L.GaussianPyramid(g_norm,int(num_levels))
        lp = L.LaplacianPyramid(g_norm,int(num_levels))
        levels = int(num_levels)
        for i in range(levels):
            cv2.imshow("Laplacian Image at level "+str(levels-i), lp[i])
            cv2.imwrite("Laplacian Image at level " + str(levels-i)+".png", lp[i])

    elif option_code == "6":
        gaussian_image_1 = input("Enter picture name to reconstruct using pyramids: ")
        num_levels = input("Enter number of levels in the pyramids (n): ")
        print("Need gaussian pyramid to build laplacian pyramid")
        reduce_kernel_1 = input("Enter separable 1-D kernel (e.g. [1/4,2/4,1/4]) : ")
        L = Blend()
        L.first_1D = np.array(eval(reduce_kernel_1))
        L.second_1D = np.transpose([L.first_1D])
        GI = cv2.imread(gaussian_image_1, 1)
        g_image = np.zeros(GI.shape)
        g_norm = cv2.normalize(GI, g_image, 0, 255, cv2.NORM_MINMAX)
        gp = L.GaussianPyramid(g_norm,int(num_levels))
        lp = L.LaplacianPyramid(g_norm,int(num_levels))
        levels = int(num_levels)
        reconstructed_pyramid,difference_pyramid = L.reconstruct(levels)
        cv2.imshow("Final reconstructed image", reconstructed_pyramid[levels-2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
