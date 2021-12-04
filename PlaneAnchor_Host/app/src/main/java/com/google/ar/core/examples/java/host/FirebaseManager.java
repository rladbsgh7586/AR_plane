/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.host;

import android.content.Context;
import android.opengl.Matrix;
import android.util.Log;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.ar.core.Anchor;
import com.google.ar.core.Plane;
import com.google.common.base.Preconditions;
import com.google.firebase.FirebaseApp;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.MutableData;
import com.google.firebase.database.Transaction;
import com.google.firebase.database.ValueEventListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageMetadata;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** A helper class to manage all communications with Firebase. */
class FirebaseManager {
    private static final String TAG =
            HelloArActivity.class.getSimpleName() + "." + FirebaseManager.class.getSimpleName();

    /** Listener for a new room code. */
    interface RoomCodeListener {

        /** Invoked when a new room code is available from Firebase. */
        void onNewRoomCode(Long newRoomCode);

        /** Invoked if a Firebase Database Error happened while fetching the room code. */
        void onError(DatabaseError error);
    }

    /** Listener for a new cloud anchor ID. */
    interface CloudAnchorIdListener {

        /** Invoked when a new cloud anchor ID is available. */
        void onNewCloudAnchorId(String cloudAnchorId);
    }

    // Names of the nodes used in the Firebase Database
    private static final String ROOT_FIREBASE_HOTSPOTS = "hotspot_list";
    private static final String ROOT_LAST_ROOM_CODE = "last_room_code";

    // Some common keys and values used when writing to the Firebase Database.
    private static final String KEY_DISPLAY_NAME = "display_name";
    private static final String KEY_ANCHOR_ID = "hosted_anchor_id";
    private static final String KEY_TIMESTAMP = "updated_at_timestamp";
    private static final String DISPLAY_NAME_VALUE = "Android EAP Sample";

    private final FirebaseApp app;
    private final DatabaseReference hotspotListRef;
    private final DatabaseReference roomCodeRef;
    private DatabaseReference currentRoomRef = null;
    private ValueEventListener currentRoomListener = null;

    private final Object databaseLock = new Object();
    private FirebaseStorage storage = null;
    private StorageReference storageImgRef = null;
    private StorageReference currentRoomImgRef = null;
    private ArrayList<String> planeID = new ArrayList<>();

    /**
     * Default constructor for the FirebaseManager.
     *
     * @param context The application context.
     */
    FirebaseManager(Context context) {
        app = FirebaseApp.initializeApp(context);
        if (app != null) {
            DatabaseReference rootRef = FirebaseDatabase.getInstance().getReference();
            storage = FirebaseStorage.getInstance();
            storageImgRef = storage.getReference().child("Image");
            Log.d("yunho-storageImgRef", storage.getReference().getName());
            Log.d("yunho-storageImgRef", storage.getReference().getPath());
            hotspotListRef = rootRef.child(ROOT_FIREBASE_HOTSPOTS);
            roomCodeRef = rootRef.child(ROOT_LAST_ROOM_CODE);

            DatabaseReference.goOnline();
        } else {
            Log.d(TAG, "Could not connect to Firebase Database!");
            hotspotListRef = null;
            roomCodeRef = null;
            storageImgRef = null;
        }
    }

    /**
     * Gets a new room code from the Firebase Database. Invokes the listener method when a new room
     * code is available.
     */
    void getNewRoomCode(RoomCodeListener listener) {
        Preconditions.checkNotNull(app, "Firebase App was null");
        roomCodeRef.runTransaction(
                new Transaction.Handler() {
                    @Override
                    public Transaction.Result doTransaction(MutableData currentData) {
                        Long nextCode = Long.valueOf(1);
                        Object currVal = currentData.getValue();
                        if (currVal != null) {
                            Long lastCode = Long.valueOf(currVal.toString());
                            nextCode = lastCode + 1;
                        }
                        currentData.setValue(nextCode);
                        return Transaction.success(currentData);
                    }

                    @Override
                    public void onComplete(DatabaseError error, boolean committed, DataSnapshot currentData) {
                        if (!committed) {
                            listener.onError(error);
                            return;
                        }
                        Long roomCode = currentData.getValue(Long.class);
                        listener.onNewRoomCode(roomCode);
                    }
                });
    }

    /** Stores the given anchor ID in the given room code. */
    void storeAnchorIdInRoom(Long roomCode, String cloudAnchorId) {
        Preconditions.checkNotNull(app, "Firebase App was null");
        DatabaseReference roomRef = hotspotListRef.child(String.valueOf(roomCode));
        currentRoomRef = hotspotListRef.child(String.valueOf(roomCode));
        currentRoomImgRef = storageImgRef.child(String.valueOf(roomCode));
        Log.d("yunho-storageRoomImgRef", "HI");
        Log.d("yunho-storageRoomImgRef", currentRoomImgRef.getName());
        Log.d("yunho-storageRoomImgRef", currentRoomImgRef.getPath());
        roomRef.child(KEY_DISPLAY_NAME).setValue(DISPLAY_NAME_VALUE);
        roomRef.child(KEY_ANCHOR_ID).setValue(cloudAnchorId);
        roomRef.child(KEY_TIMESTAMP).setValue(System.currentTimeMillis());
    }

    /**
     * Registers a new listener for the given room code. The listener is invoked whenever the data for
     * the room code is changed.
     */
    void registerNewListenerForRoom(Long roomCode, CloudAnchorIdListener listener) {
        Preconditions.checkNotNull(app, "Firebase App was null");
        clearRoomListener();
        currentRoomRef = hotspotListRef.child(String.valueOf(roomCode));
        currentRoomListener =
                new ValueEventListener() {
                    @Override
                    public void onDataChange(DataSnapshot dataSnapshot) {
                        Object valObj = dataSnapshot.child(KEY_ANCHOR_ID).getValue();
                        if (valObj != null) {
                            String anchorId = String.valueOf(valObj);
                            if (!anchorId.isEmpty()) {
                                listener.onNewCloudAnchorId(anchorId);
                            }
                        }
                    }

                    @Override
                    public void onCancelled(DatabaseError databaseError) {
                        Log.w(TAG, "The Firebase operation was cancelled.", databaseError.toException());
                    }
                };
        currentRoomRef.addValueEventListener(currentRoomListener);
    }

    public void readData(DatabaseReference ref, final HelloArActivity.myCallBack callBack){
        ArrayList<PlaneAnchor> planeAnchors = new ArrayList<PlaneAnchor>();
        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                for(DataSnapshot snapshot: dataSnapshot.getChildren()){
                    PlaneAnchor plane = snapshot.getValue(PlaneAnchor.class);

                    Log.d("yunho", snapshot.child("transformation_matrix").getValue().getClass().toString());
                    List<Double> matrixList = (List<Double>) snapshot.child("transformation_matrix").getValue();
                    Log.d("yunho",Arrays.toString(matrixList.toArray()));
                    float[] matrixArray = new float[matrixList.size()];
                    int i = 0;

                    for (Object d : matrixList) {
                        Log.d("yunho",String.valueOf(d));
                        Log.d("yunho", d.getClass().toString());
                        if(d.getClass() == Double.class){
                            matrixArray[i++] = (float)((double)d); // Or whatever default you want.
                        }
                        else if(d.getClass() == Long.class){
                            matrixArray[i++] = (float)((long)d); // Or whatever default you want.
                        }
                        else{
                            Log.d("yunho", "error");
                        }

                    }
                    plane.setTransformationMatrix(matrixArray);
                    planeAnchors.add(plane);
                    Log.d("yunho-2", Boolean.toString(planeAnchors.isEmpty()));
                }
                callBack.onSuccess(planeAnchors);
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });
    }

    DatabaseReference getPlaneAnchorsRef(Long roomCode){
        return hotspotListRef.child(String.valueOf(roomCode)).child("plane_anchors");
    }

    void uploadImage(byte[] file, String fileName, float[] invModelViewMatrix, String timestamp){
        Log.d("yunho-name", currentRoomImgRef.getName());
        Log.d("yunho-name", currentRoomImgRef.getPath());
        StorageMetadata metadata = new StorageMetadata.Builder()
                .setCustomMetadata("inverseModelViewMatrix", Arrays.toString(invModelViewMatrix))
                .setCustomMetadata("timestamp", timestamp)
                .build();
        UploadTask uploadTask = currentRoomImgRef.child(fileName+".jpg").putBytes(file);
        uploadTask.addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {
            }
        }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
            @Override
            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                currentRoomImgRef.child(fileName+".jpg").updateMetadata(metadata);
                // taskSnapshot.getMetadata() contains file metadata such as size, content-type, etc.
                // ...
            }
        });
    };

    void uploadPCD(byte[] file, String fileName){
        UploadTask uploadTask = currentRoomImgRef.child(fileName+"_pcd.txt").putBytes(file);
        uploadTask.addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {
            }
        }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
            @Override
            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                // taskSnapshot.getMetadata() contains file metadata such as size, content-type, etc.
                // ...
            }
        });
    };

    float[] planeMatrix = new float[16];
    float[] anchorMatrix = new float[16];
    float[] anchorInverseMatrix = new float[16];
    float[] planeAnchorMatrix = new float[16];

//    void uploadPlane(Plane plane, Anchor anchor, int num){
//        Log.d("yunho",Boolean.toString(planeID.contains(plane.toString())));
//        if(!planeID.contains(plane.toString())){
//            Log.d("yunho",plane.toString());
//            planeID.add(plane.toString());
//
//            anchor.getPose().toMatrix(anchorMatrix, 0);
//            plane.getCenterPose().toMatrix(planeMatrix, 0);
//            Matrix.invertM(anchorInverseMatrix, 0, anchorMatrix,0);
//            Matrix.multiplyMM(planeAnchorMatrix, 0, anchorInverseMatrix, 0, planeMatrix, 0);
//            Double[] doublePlaneAnchorMatrix = new Double[planeAnchorMatrix.length];
//            for(int i = 0; i < planeAnchorMatrix.length; i++){
//                doublePlaneAnchorMatrix[i] = (double)planeAnchorMatrix[i];
//            }
//
//            List<Double> list = Arrays.asList(doublePlaneAnchorMatrix);
//            DatabaseReference ref = currentRoomRef.child("plane_anchors").child("plane"+Integer.toString(planeID.size())).getRef();
//            ref.child("transformation_matrix").setValue(list);
//            ref.child("width").setValue(plane.getExtentX() * 1000);
//            ref.child("height").setValue(plane.getExtentZ() * 1000);
//        }
////
//    };

    PlaneAnchor uploadPlane(Plane plane, Anchor anchor, int num){
        PlaneAnchor planeAnchor = null;
        if(!planeID.contains(plane.toString())){
            Log.d("yunho",plane.toString());
            planeID.add(plane.toString());

            anchor.getPose().toMatrix(anchorMatrix, 0);
            plane.getCenterPose().toMatrix(planeMatrix, 0);
            Matrix.invertM(anchorInverseMatrix, 0, anchorMatrix,0);
            Matrix.multiplyMM(planeAnchorMatrix, 0, anchorInverseMatrix, 0, planeMatrix, 0);
            Double[] doublePlaneAnchorMatrix = new Double[planeAnchorMatrix.length];
            for(int i = 0; i < planeAnchorMatrix.length; i++){
                doublePlaneAnchorMatrix[i] = (double)planeAnchorMatrix[i];
            }

            List<Double> list = Arrays.asList(doublePlaneAnchorMatrix);
            DatabaseReference ref = currentRoomRef.child("plane_anchors").child("plane"+Integer.toString(planeID.size())).getRef();
            ref.child("transformation_matrix").setValue(list);
            ref.child("width").setValue(plane.getExtentX() * 1000);
            ref.child("height").setValue(plane.getExtentZ() * 1000);

            planeAnchor = new PlaneAnchor(planeAnchorMatrix, (int)(plane.getExtentX() * 1000), (int)(plane.getExtentZ() * 1000));
        }
        return planeAnchor;
    };

    /**
     * Resets the current room listener registered using {@link #registerNewListenerForRoom(Long,
     * CloudAnchorIdListener)}.
     */
    void clearRoomListener() {
        if (currentRoomListener != null && currentRoomRef != null) {
            currentRoomRef.removeEventListener(currentRoomListener);
            currentRoomListener = null;
            currentRoomRef = null;
        }
    }
}
