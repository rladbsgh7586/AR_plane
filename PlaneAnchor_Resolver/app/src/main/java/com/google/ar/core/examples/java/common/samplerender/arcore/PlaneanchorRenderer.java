/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ar.core.examples.java.common.samplerender.arcore;

import android.opengl.EGLConfig;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.Log;

import com.google.ar.core.Anchor;
import com.google.ar.core.Camera;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.samplerender.IndexBuffer;
import com.google.ar.core.examples.java.common.samplerender.Mesh;
import com.google.ar.core.examples.java.common.samplerender.SampleRender;
import com.google.ar.core.examples.java.common.samplerender.Shader;
import com.google.ar.core.examples.java.common.samplerender.Shader.BlendFactor;
import com.google.ar.core.examples.java.common.samplerender.Square;
import com.google.ar.core.examples.java.common.samplerender.Texture;
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.microedition.khronos.opengles.GL10;

/** Renders the detected AR planes. */
public class PlaneanchorRenderer {
    // Shader names.
    private Square mSquare;

    private static final String VERTEX_SHADER_NAME = "shaders/plane_anchor.vert";
    private static final String FRAGMENT_SHADER_NAME = "shaders/plane_anchor.frag";

    private static final int COORDS_PER_VERTEX = 3; // x, y, z

    private final Shader shader;

    // Temporary lists/matrices allocated here to reduce number of allocations for each frame.
    private final float[] viewMatrix = new float[16];
    private final float[] modelViewMatrix = new float[16];
    private final float[] modelViewProjectionMatrix = new float[16];

    public PlaneanchorRenderer(SampleRender render) throws IOException {
        shader =
                Shader.createFromAssets(render, VERTEX_SHADER_NAME, FRAGMENT_SHADER_NAME, /*defines=*/ null)
                        .setBlend(
                                BlendFactor.DST_ALPHA, // RGB (src)
                                BlendFactor.ONE, // RGB (dest)
                                BlendFactor.ZERO, // ALPHA (src)
                                BlendFactor.ONE_MINUS_SRC_ALPHA) // ALPHA (dest)
                        .setDepthWrite(false);
    }

    /**
     * Draws the collection of tracked planes, with closer planes hiding more distant ones.
     *
     * @param allPlanes The collection of planes to draw.
     * @param cameraPose The pose of the camera, as returned by {@link Camera#getPose()}
     * @param cameraProjection The projection matrix, as returned by {@link
     *     Camera#getProjectionMatrix(float[], int, float, float)}
     */
    public void drawPlaneAnchors(
            SampleRender render, float[] anchorMatrix, Pose cameraPose, float[] cameraProjection, Mesh mesh, Texture texture, float[] color, float[] border_color) {
//        shader.setTexture("u_Texture", texture);
        cameraPose.inverse().toMatrix(viewMatrix, 0);
        Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, anchorMatrix, 0);
        Matrix.multiplyMM(modelViewProjectionMatrix, 0, cameraProjection, 0, modelViewMatrix, 0);
        // Populate the shader uniforms for this frame.
        shader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);
        shader.setVec4("color", color);
        shader.setVec4("border_color", border_color);

        render.draw(mesh, shader);
    }
}
