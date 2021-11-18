package com.google.ar.core.examples.java.common.samplerender;

import android.opengl.GLES20;
import android.util.Log;

import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneanchorRenderer;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

public class Square {

    private FloatBuffer vertexBuffer;
    private FloatBuffer textureCordinateBuffer;
    private IntBuffer indexBuffer;

    static final int COORDS_PER_VERTEX = 3;
    static float squareCoords[] = new float[12];
    static float textureCoords[] = {1, 1, 1, 0, 0, 0, 0, 1};
    private int drawIndex[] = { 0, 1, 3, 1, 2, 3 }; // order to draw vertices

    public Square(int width, int height) {
        for(int i = 0; i < squareCoords.length; i++){
            switch(i%3){
                // x
                case 0:
                    squareCoords[i] = (float)width / 2000 ;
//                    squareCoords[i] = 0.5f;
                    break;

                case 1:
                    squareCoords[i] = 0.0f;
//                    squareCoords[i] = 0.5f;
                    break;

                case 2:
                    squareCoords[i] = (float)height / 2000;
//                    squareCoords[i] = (float)height / 2;
                    break;

            }
        }
        squareCoords[5] = squareCoords[5] * -1;
        squareCoords[6] = squareCoords[6] * -1;
        squareCoords[8] = squareCoords[8] * -1;
        squareCoords[9] = squareCoords[9] * -1;

        // initialize vertex byte buffer for shape coordinates
        ByteBuffer bb = ByteBuffer.allocateDirect(
                // (# of coordinate values * 4 bytes per float)
                squareCoords.length * 4);
        bb.order(ByteOrder.nativeOrder());
        vertexBuffer = bb.asFloatBuffer();
        vertexBuffer.put(squareCoords);
        vertexBuffer.position(0);

        ByteBuffer bb1 = ByteBuffer.allocateDirect(
                // (# of coordinate values * 4 bytes per float)
                textureCoords.length * 4);
        bb1.order(ByteOrder.nativeOrder());
        textureCordinateBuffer = bb1.asFloatBuffer();
        textureCordinateBuffer.put(textureCoords);
        textureCordinateBuffer.position(0);

        // initialize byte buffer for the draw list
        ByteBuffer dlb = ByteBuffer.allocateDirect(
                // (# of coordinate values * 4 bytes per int)
                drawIndex.length * 4);
        dlb.order(ByteOrder.nativeOrder());
        indexBuffer = dlb.asIntBuffer();
        indexBuffer.put(drawIndex);
        indexBuffer.position(0);
    }

    public FloatBuffer getVertexBuffer(){ return vertexBuffer; }

    public IntBuffer getIndexBuffer(){
        return indexBuffer;
    }

    public FloatBuffer getTextureCoordinateBuffer(){return textureCordinateBuffer;}
}
