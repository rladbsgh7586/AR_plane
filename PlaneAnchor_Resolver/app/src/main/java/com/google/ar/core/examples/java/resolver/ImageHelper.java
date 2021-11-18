package com.google.ar.core.examples.java.resolver;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class ImageHelper {
    public static byte[] imageToByteArray(Image image) {
        byte[] data = null;
        if (image.getFormat() == ImageFormat.JPEG) {
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer buffer = planes[0].getBuffer();
            data = new byte[buffer.capacity()];
            buffer.get(data);
            return data;
        } else if (image.getFormat() == ImageFormat.YUV_420_888) {
//            byte[] rotated_nv21 = rotateNV21(YUV_420_888toNV21(image), image.getWidth(), image.getHeight(), 90);
//            data = NV21toJPEG(
//                    rotated_nv21,
//                    image.getHeight(), image.getWidth());
            data = NV21toJPEG(
                    rotateYUV420ToNV21(image),
                    image.getHeight(), image.getWidth());
//            data = NV21toJPEG(
//                    YUV_420_888toNV21(image),
//                    image.getWidth(), image.getHeight());
        }
        return data;
    }

    private static byte[] YUV_420_888toNV21(Image imgYUV420) {
//        byte[] nv21;
//        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
//        ByteBuffer vuBuffer = image.getPlanes()[2].getBuffer();
//
//        int ySize = yBuffer.remaining();
//        int vuSize = vuBuffer.remaining();
//
//        nv21 = new byte[ySize + vuSize];
//
//        yBuffer.get(nv21, 0, ySize);
//        vuBuffer.get(nv21, ySize, vuSize);
//
//        return nv21;
        assert(imgYUV420.getFormat() == ImageFormat.YUV_420_888);

        byte[] rez = new byte[imgYUV420.getWidth() * imgYUV420.getHeight() * 3 / 2];
        ByteBuffer buffer0 = imgYUV420.getPlanes()[0].getBuffer();
        ByteBuffer buffer1 = imgYUV420.getPlanes()[2].getBuffer();
        ByteBuffer buffer2 = imgYUV420.getPlanes()[1].getBuffer();

        int n = 0;
        assert(imgYUV420.getPlanes()[0].getPixelStride() == 1);
        for (int row = 0; row < imgYUV420.getHeight(); row++) {
            for (int col = 0; col < imgYUV420.getWidth(); col++) {
                rez[n++] = buffer0.get();
            }
        }
        assert(imgYUV420.getPlanes()[2].getPixelStride() == imgYUV420.getPlanes()[1].getPixelStride());
        int stride = imgYUV420.getPlanes()[1].getPixelStride();
        for (int row = 0; row < imgYUV420.getHeight(); row += 2) {
            for (int col = 0; col < imgYUV420.getWidth(); col += 2) {
                rez[n++] = buffer1.get();
                rez[n++] = buffer2.get();
                for (int skip = 1; skip < stride; skip++) {
                    if (buffer1.remaining() > 0) {
                        buffer1.get();
                    }
                    if (buffer2.remaining() > 0) {
                        buffer2.get();
                    }
                }
            }
        }

        return rez;
    }

    private static byte[] rotateYUV420ToNV21(Image imgYUV420) {

        byte[] rez = new byte[imgYUV420.getWidth() * imgYUV420.getHeight() * 3 / 2];
        ByteBuffer buffer0 = imgYUV420.getPlanes()[0].getBuffer();
        ByteBuffer buffer1 = imgYUV420.getPlanes()[2].getBuffer();
        ByteBuffer buffer2 = imgYUV420.getPlanes()[1].getBuffer();

        int width = imgYUV420.getHeight();
        assert(imgYUV420.getPlanes()[0].getPixelStride() == 1);
        for (int row = imgYUV420.getHeight()-1; row >=0; row--) {
            for (int col = 0; col < imgYUV420.getWidth(); col++) {
                rez[col*width+row] = buffer0.get();
            }
        }
        int uv_offset = imgYUV420.getWidth()*imgYUV420.getHeight();
        assert(imgYUV420.getPlanes()[2].getPixelStride() == imgYUV420.getPlanes()[1].getPixelStride());
        int stride = imgYUV420.getPlanes()[1].getPixelStride();
        for (int row = imgYUV420.getHeight() - 2; row >= 0; row -= 2) {
            for (int col = 0; col < imgYUV420.getWidth(); col += 2) {
                rez[uv_offset+col/2*width+row] = buffer1.get();
                rez[uv_offset+col/2*width+row+1] = buffer2.get();
                for (int skip = 1; skip < stride; skip++) {
                    if (buffer1.remaining() > 0) {
                        buffer1.get();
                    }
                    if (buffer2.remaining() > 0) {
                        buffer2.get();
                    }
                }
            }
        }

        return rez;
    }

    private static byte[] NV21toJPEG(byte[] nv21, int width, int height) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
//        yuv.compressToJpeg(new Rect(0, 0, width, height), 100, out);
        yuv.compressToJpeg(new Rect(0, 0, width, (int)(width*3/4)), 100, out);
        return out.toByteArray();
    }

    public static byte[] rotateNV21(final byte[] yuv,
                                            final int width,
                                            final int height,
                                            final int rotation)
    {
        if (rotation == 0) return yuv;
        if (rotation % 90 != 0 || rotation < 0 || rotation > 270) {
            throw new IllegalArgumentException("0 <= rotation < 360, rotation % 90 == 0");
        }

        final byte[]  output    = new byte[yuv.length];
        final int     frameSize = width * height;
        final boolean swap      = rotation % 180 != 0;
        final boolean xflip     = rotation % 270 != 0;
        final boolean yflip     = rotation >= 180;

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                final int yIn = j * width + i;
                final int uIn = frameSize + (j >> 1) * width + (i & ~1);
                final int vIn = uIn       + 1;

                final int wOut     = swap  ? height              : width;
                final int hOut     = swap  ? width               : height;
                final int iSwapped = swap  ? j                   : i;
                final int jSwapped = swap  ? i                   : j;
                final int iOut     = xflip ? wOut - iSwapped - 1 : iSwapped;
                final int jOut     = yflip ? hOut - jSwapped - 1 : jSwapped;

                final int yOut = jOut * wOut + iOut;
                final int uOut = frameSize + (jOut >> 1) * wOut + (iOut & ~1);
                final int vOut = uOut + 1;

                output[yOut] = (byte)(0xff & yuv[yIn]);
                output[uOut] = (byte)(0xff & yuv[uIn]);
                output[vOut] = (byte)(0xff & yuv[vIn]);
            }
        }
        return output;
    }

    public static void writeFrame(String fileName, byte[] data) {
        try {
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(fileName));
            bos.write(data);
            bos.flush();
            bos.close();
            Log.d("Image", "" + data.length + " bytes have been written to " + fileName + ".jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
