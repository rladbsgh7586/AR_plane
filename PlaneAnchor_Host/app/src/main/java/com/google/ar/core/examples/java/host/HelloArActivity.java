/*
 * Copyright 2017 Google LLC
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

import android.Manifest;
import android.accounts.AccountManager;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.media.Image;
import android.net.Uri;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.GuardedBy;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.MotionEventCompat;
import androidx.fragment.app.DialogFragment;
import androidx.loader.content.CursorLoader;

import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Config.InstantPlacementMode;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.LightEstimate;
import com.google.ar.core.Plane;
import com.google.ar.core.PlaybackStatus;
import com.google.ar.core.Point;
import com.google.ar.core.Point.OrientationMode;
import com.google.ar.core.PointCloud;
import com.google.ar.core.RecordingConfig;
import com.google.ar.core.RecordingStatus;
import com.google.ar.core.Session;
import com.google.ar.core.Track;
import com.google.ar.core.TrackData;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingFailureReason;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.common.samplerender.arcore.TextRenderer;
import com.google.ar.core.examples.java.host.PrivacyNoticeDialogFragment.HostResolveListener;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DepthSettings;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.InstantPlacementSettings;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TapHelper;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.examples.java.common.samplerender.Framebuffer;
import com.google.ar.core.examples.java.common.samplerender.GLError;
import com.google.ar.core.examples.java.common.samplerender.IndexBuffer;
import com.google.ar.core.examples.java.common.samplerender.Mesh;
import com.google.ar.core.examples.java.common.samplerender.SampleRender;
import com.google.ar.core.examples.java.common.samplerender.Shader;
import com.google.ar.core.examples.java.common.samplerender.Square;
import com.google.ar.core.examples.java.common.samplerender.Texture;
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer;
import com.google.ar.core.examples.java.common.samplerender.arcore.BackgroundRenderer;
import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneRenderer;
import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneanchorRenderer;
import com.google.ar.core.examples.java.common.samplerender.arcore.SpecularCubemapFilter;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.PlaybackFailedException;
import com.google.ar.core.exceptions.RecordingFailedException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.common.base.Preconditions;
import com.google.firebase.database.DatabaseError;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore API. The application will display any detected planes and will allow the user to tap on a
 * plane to place a 3D model.
 */
public class HelloArActivity extends AppCompatActivity implements SampleRender.Renderer, PrivacyNoticeDialogFragment.NoticeDialogListener {

  private static final String TAG = HelloArActivity.class.getSimpleName();

  private static final String SEARCHING_PLANE_MESSAGE = "Searching for surfaces...";
  private static final String HOSTING_PLANE_MESSAGE = "Hosting plane anchors...";
  private static final String WAITING_FOR_TAP_MESSAGE = "Tap on a surface to place an object.";

  private static final int PERMISSIONS_REQUEST_CAMERA = 0;
  private static final int PERMISSIONS_REQUEST_WRITE = 1;

  private String serverIP = "115.145.175.42";
  private int serverPort = 7586;
  private Long recentRoomCode = 0L;

  // See the definition of updateSphericalHarmonicsCoefficients for an explanation of these
  // constants.
  private static final float[] sphericalHarmonicFactors = {
    0.282095f,
    -0.325735f,
    0.325735f,
    -0.325735f,
    0.273137f,
    -0.273137f,
    0.078848f,
    -0.273137f,
    0.136569f,
  };

  private static final float Z_NEAR = 0.1f;
  private static final float Z_FAR = 100f;

  private static final int CUBEMAP_RESOLUTION = 16;
  private static final int CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES = 32;
  private static final int PLANE_TEXT_WIDTH = 256;
  private static final int PLANE_TEXT_HEIGHT = 128;

  class PlaneRender{
    Mesh planeMesh;
    float[] planeMatrix;
    Texture planeName;

    PlaneRender(Mesh mesh, float[] matrix, Texture name){
      this.planeMesh = mesh;
      this.planeMatrix = matrix;
      this.planeName = name;
    }

    PlaneRender(){

    }

    Mesh getPlaneMesh(){
      return this.planeMesh;
    }

    float[] getPlaneMatrix(){
      return this.planeMatrix;
    }

    Texture getPlaneTexture(){
      return this.planeName;
    }

  }

  ArrayList<PlaneAnchor> planeAnchors = new ArrayList<PlaneAnchor>();
  ArrayList<PlaneRender> planeRenders = new ArrayList<PlaneRender>();

  // Rendering. The Renderers are created here, and initialized when the GL surface is created.
  private GLSurfaceView surfaceView;

  private boolean installRequested;
  private boolean shouldMakeRenders = false;

  private Session session;
  public SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
  private DisplayRotationHelper displayRotationHelper;
  private final TrackingStateHelper trackingStateHelper = new TrackingStateHelper(this);
  private TapHelper tapHelper;
  private SampleRender render;

  private PlaneRenderer planeRenderer;
  private TextRenderer textRenderer;
  private TextRenderer cloudAnchorTextRenderer;
  private PlaneanchorRenderer planeanchorRenderer;
  private BackgroundRenderer backgroundRenderer;
  private Framebuffer virtualSceneFramebuffer;
  private RemoteServerManager remoteServerManager;
  private boolean hasSetTextureNames = false;
  private boolean resolveCalled = false;

  private final DepthSettings depthSettings = new DepthSettings();
  private boolean[] depthSettingsMenuDialogCheckboxes = new boolean[2];

  private final InstantPlacementSettings instantPlacementSettings = new InstantPlacementSettings();
  private boolean[] instantPlacementSettingsMenuDialogCheckboxes = new boolean[1];
  // Assumed distance from the device camera to the surface on which user will try to place objects.
  // This value affects the apparent scale of objects while the tracking method of the
  // Instant Placement point is SCREENSPACE_WITH_APPROXIMATE_DISTANCE.
  // Values in the [0.2, 2.0] meter range are a good choice for most AR experiences. Use lower
  // values for AR experiences where users are expected to place objects on surfaces close to the
  // camera. Use larger values for experiences where the user will likely be standing and trying to
  // place an object on the ground or floor in front of them.
  private static final float APPROXIMATE_DISTANCE_METERS = 2.0f;

  // Point Cloud
  private VertexBuffer pointCloudVertexBuffer;
  private Mesh pointCloudMesh;
  private Shader pointCloudShader;
  // Keep track of the last point cloud rendered to avoid updating the VBO if point cloud
  // was not changed.  Do this using the timestamp since we can't compare PointCloud objects.
  private long lastPointCloudTimestamp = 0;

  // Virtual object (ARCore pawn)
  private Mesh virtualObjectMesh;
  private Mesh anchorMesh;
  private Mesh planeMeshA;
  private Texture textureA;
  private Texture textureB;
  private Texture cloudAnchorTextTexture;
  private float[] anchorColor = {0.0f, 1.0f, 0.0f, 0.7f};
  private float[] planeAnchorColor = {1.0f, 0.0f, 0.0f, 0.3f};
  private float[] borderColorA = {1.0f, 1.0f, 1.0f, 0.0f};
  private float[] borderColorB = {1.0f, 1.0f, 1.0f, 0.0f};
  private float[] planeAnchorColorB = {0.0f, 0.0f, 1.0f, 0.3f};
  private Shader virtualObjectShader;
  private final ArrayList<Anchor> anchors = new ArrayList<>();

  // Environmental HDR
  private Texture dfgTexture;
  private SpecularCubemapFilter cubemapFilter;

  // Temporary matrix allocated here to reduce number of allocations for each frame.
  private final float[] modelMatrix = new float[16];
  private final float[] viewMatrix = new float[16];
  private final float[] projectionMatrix = new float[16];
  private final float[] modelViewMatrix = new float[16]; // view x model
  private final float[] modelViewProjectionMatrix = new float[16]; // projection x view x model
  private final float[] ViewProjectionMatrix = new float[16]; // view x model
  private final float[] invModelViewMatrix = new float[16]; // projection x view x model
  private final float[] sphericalHarmonicsCoefficients = new float[9 * 3];
  private final float[] viewInverseMatrix = new float[16];
  private final float[] worldLightDirection = {0.0f, 0.0f, 0.0f, 0.0f};
  private final float[] viewLightDirection = new float[4]; // view x world light direction

  private Set<Integer> previousKeyFrameIds = new HashSet<Integer>(0);
  private int keyFrameThreshold = 10;
  private int keyFrameCount = 0;
  private int counter = 0;
  private long scenario = 0;
  private ViewHandler viewHandler;

  public enum RecordType{
    HOST,
    RESOLVE
  }

  private RecordType recordType = RecordType.HOST;

  public enum AppState {
    Idle,
    Recording,
    Playingback
  }

  private enum HostResolveMode {
    NONE,
    HOSTING,
    HOSTINGDONE,
    PLANEHOSTING,
    RESOLVING,
  }

  private enum GroundTruthMode {
    TRUE,
    FALSE
  }

  String method = "arcore";

  // Locks needed for synchronization
  private final Object singleTapLock = new Object();
  private final Object anchorLock = new Object();

  private GestureDetector gestureDetector;
  private final SnackbarHelper snackbarHelper = new SnackbarHelper();
  private Button hostButton;
  private Button resolveButton;
  private Button scenarioButton;
  private Button imageHostButton;
  private TextView roomCodeText;
  private SharedPreferences sharedPreferences;
  private static final String PREFERENCE_FILE_KEY = "allow_sharing_images";
  private static final String ALLOW_SHARE_IMAGES_KEY = "ALLOW_SHARE_IMAGES";

  @GuardedBy("singleTapLock")
  private MotionEvent queuedSingleTap;


  @GuardedBy("anchorLock")
  private Anchor anchor;

  // Cloud Anchor Components.
  private FirebaseManager firebaseManager;
  private final CloudAnchorManager cloudManager = new CloudAnchorManager();
  private HostResolveMode currentMode;
  private RoomCodeAndCloudAnchorIdListener hostListener;
  private GroundTruthRecorder groundTruthRecorder;

  // Tracks app's specific state changes.
  private AppState appState = AppState.Idle;
  private GroundTruthMode gtMode = GroundTruthMode.FALSE;
  private int REQUEST_MP4_SELECTOR = 1;
  private int STORE_ARCORE_PLANES = 0;

  private String recording_name = "indoor_recording.mp4";

  private UUID trackUUID = UUID.fromString("9584c0cc-7b88-4796-a7ad-e394a7f55b29");
  private RecordingConfig recordingConfig;
  private float[] touchCoordinate = new float[2];
  private float[] imageHostingCoordinate = new float[] {-1, -1};

  private boolean anchorTouched = false;
  private boolean imageHostingTouched = false;
  private boolean doneTouched = false;

  private TextView message_text;
//  MainHandler mainHandler;
//
//  class MainHandler extends Handler {
//    @Override
//    public void handleMessage(@NonNull Message msg){
//      super.handleMessage(msg);
//
//      Bundle bundle = msg.getData();
//      message_text.setText(bundle.getString("data"));
//    }
//  }

  private void updateRecordButton() {
    View buttonView = findViewById(R.id.record_button);
    Button button = (Button) buttonView;

    switch (appState) {
      case Idle:
        button.setText("Record");
        button.setVisibility(View.VISIBLE);
        break;
      case Recording:
        button.setText("Stop");
        button.setVisibility(View.VISIBLE);
        break;
      case Playingback:
        button.setVisibility(View.INVISIBLE);
        break;
    }
  }

  private void updatePlaybackButton() {
    View buttonView = findViewById(R.id.playback_button);
    Button button = (Button)buttonView;

    switch (appState) {

      // The app is neither recording nor playing back. The "Playback" button is visible.
      case Idle:
        button.setText("Playback");
        button.setVisibility(View.VISIBLE);
        break;

      // While playing back, the "Playback" button is visible and says "Stop".
      case Playingback:
        button.setText("Stop");
        button.setVisibility(View.VISIBLE);
        break;

      // During recording, the "Playback" button is not visible.
      case Recording:
        button.setVisibility(View.INVISIBLE);
        break;
    }
  }

  public void onClickRecord(View view) {
    Log.d(TAG, "onClickRecord");

    // Check the app's internal state and switch to the new state if needed.
    switch (appState) {
      // If the app is not recording, begin recording.
      case Idle: {
        boolean hasStarted = startRecording();
        Log.d(TAG, String.format("onClickRecord start: hasStarted %b", hasStarted));

        if (hasStarted)
          appState = AppState.Recording;

        break;
      }

      // If the app is recording, stop recording.
      case Recording: {
        boolean hasStopped = stopRecording();
        Log.d(TAG, String.format("onClickRecord stop: hasStopped %b", hasStopped));

        if (hasStopped)
          appState = AppState.Idle;

        break;
      }

      default:
        // Do nothing.
        break;
    }
    updateRecordButton();
    updatePlaybackButton();
  }

  public void onClickPlayback(View view) {
    Log.d(TAG, "onClickPlayback");

    switch (appState) {

      // If the app is not playing back, open the file picker.
      case Idle: {
        boolean hasStarted = selectFileToPlayback();
        Log.d(TAG, String.format("onClickPlayback start: selectFileToPlayback %b", hasStarted));
        break;
      }

      // If the app is playing back, stop playing back.
      case Playingback: {
        boolean hasStopped = stopPlayingback();
        Log.d(TAG, String.format("onClickPlayback stop: hasStopped %b", hasStopped));
        break;
      }

      default:
        // Recording - do nothing.
        break;
    }

    // Update the UI for the "Record" and "Playback" buttons.
    updateRecordButton();
    updatePlaybackButton();
  }

  private boolean selectFileToPlayback() {
    // Start file selection from Movies directory.
    // Android 10 and above requires VOLUME_EXTERNAL_PRIMARY to write to MediaStore.
    Uri videoCollection;
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      videoCollection = MediaStore.Video.Media.getContentUri(
              MediaStore.VOLUME_EXTERNAL_PRIMARY);
    } else {
      videoCollection = MediaStore.Video.Media.EXTERNAL_CONTENT_URI;
    }

    // Create an Intent to select a file.
    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);

    // Add file filters such as the MIME type, the default directory and the file category.
    intent.setType(MP4_VIDEO_MIME_TYPE); // Only select *.mp4 files
    intent.putExtra(DocumentsContract.EXTRA_INITIAL_URI, videoCollection); // Set default directory
    intent.addCategory(Intent.CATEGORY_OPENABLE); // Must be files that can be opened

    this.startActivityForResult(intent, REQUEST_MP4_SELECTOR);

    return true;
  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    // Check request status. Log an error if the selection fails.
    if (resultCode != android.app.Activity.RESULT_OK || requestCode != REQUEST_MP4_SELECTOR) {
      Log.e(TAG, "onActivityResult select file failed");
      return;
    }

    Uri mp4Uri = data.getData();
    extractScenarioFromFilePath(mp4Uri);
    Log.d(TAG, String.format("onActivityResult result is %s", mp4Uri));

    // Copy to app internal storage to get a file path.
    String localFilePath = copyToInternalFilePath(mp4Uri);

    // Begin playback.
    startPlayingback(localFilePath);
  }

  private void extractScenarioFromFilePath(Uri uri){
    String[] mp4Split = uri.getPath().split("[.]");
    String[] uriSplit = mp4Split[0].split("[(]");
    String intStr = uriSplit[0].replaceAll("[^0-9]","");
    scenario = Integer.parseInt(intStr);
  }

  private String copyToInternalFilePath(Uri contentUri) {
    // Create a file path in the app's internal storage.
    String tempPlaybackFilePath = new File(this.getExternalFilesDir(null), "temp-playback.mp4").getAbsolutePath();

    // Copy the binary content from contentUri to tempPlaybackFilePath.
    try (InputStream inputStream = this.getContentResolver().openInputStream(contentUri);
         java.io.OutputStream tempOutputFileStream = new java.io.FileOutputStream(tempPlaybackFilePath)) {

      byte[] buffer = new byte[1024 * 1024]; // 1MB
      int bytesRead = inputStream.read(buffer);
      while (bytesRead != -1) {
        tempOutputFileStream.write(buffer, 0, bytesRead);
        bytesRead = inputStream.read(buffer);
      }

    } catch (java.io.FileNotFoundException e) {
      Log.e(TAG, "copyToInternalFilePath FileNotFoundException", e);
      return null;
    } catch (IOException e) {
      Log.e(TAG, "copyToInternalFilePath IOException", e);
      return null;
    }

    // Return the absolute file path of the copied file.
    return tempPlaybackFilePath;
  }

  private boolean startPlayingback(String mp4FilePath) {
    lastPointCloudTimestamp = 0;
    if (mp4FilePath == null)
      return false;

    Log.d(TAG, "startPlayingback at:" + mp4FilePath);

    pauseARCoreSession();
    createSessionNoResume();
    try {
      session.setPlaybackDataset(mp4FilePath);
    } catch (PlaybackFailedException e) {
      Log.e(TAG, "startPlayingback - setPlaybackDataset failed", e);
    }

    // The session's camera texture name becomes invalid when the
    // ARCore session is set to play back.
    // Workaround: Reset the Texture to start Playback
    // so it doesn't crashes with AR_ERROR_TEXTURE_NOT_SET.
    hasSetTextureNames = false;

    boolean canResume = resumeARCoreSession();
    if (!canResume)
      return false;

    PlaybackStatus playbackStatus = session.getPlaybackStatus();
    Log.d(TAG, String.format("startPlayingback - playbackStatus %s", playbackStatus));


    if (playbackStatus != PlaybackStatus.OK) { // Correctness check
      return false;
    }

    appState = AppState.Playingback;
    updateRecordButton();
    updatePlaybackButton();

    return true;
  }

  private boolean stopPlayingback() {
    // Correctness check, only stop playing back when the app is playing back.
    if (appState != AppState.Playingback)
      return false;

    pauseARCoreSession();
    createSessionNoResume();

    // Close the current session and create a new session.
    boolean canResume = resumeARCoreSession();
    if (!canResume)
      return false;

    // Reset appState to Idle, and update the "Record" and "Playback" buttons.
    appState = AppState.Idle;
    updateRecordButton();
    updatePlaybackButton();

    return true;
  }

  private boolean startRecording() {
    String mp4FilePath = createMp4File();
    if (mp4FilePath == null)
      return false;

    Log.d(TAG, "startRecording at: " + mp4FilePath);

    pauseARCoreSession();

    if(recordType == RecordType.HOST){
      Track track = new Track(session).setId(trackUUID);
      // Configure the ARCore session to start recording.
      recordingConfig = new RecordingConfig(session)
              .setMp4DatasetFilePath(mp4FilePath)
              .setAutoStopOnPause(true)
              .addTrack(track);
    }
    else{
      recordingConfig = new RecordingConfig(session)
              .setMp4DatasetFilePath(mp4FilePath)
              .setAutoStopOnPause(true);
    }




    try {
      // Prepare the session for recording, but do not start recording yet.
      session.startRecording(recordingConfig);
      if(gtMode == GroundTruthMode.TRUE) {
        groundTruthRecorder.recordBagStart(recordType + String.valueOf(scenario));
      }
    } catch (RecordingFailedException e) {
      Log.e(TAG, "startRecording - Failed to prepare to start recording", e);
      return false;
    }

    boolean canResume = resumeARCoreSession();
    if (!canResume)
      return false;


    // Correctness checking: check the ARCore session's RecordingState.
    RecordingStatus recordingStatus = session.getRecordingStatus();
    Log.d(TAG, String.format("startRecording - recordingStatus %s", recordingStatus));
    return recordingStatus == RecordingStatus.OK;
  }

  private boolean stopRecording() {
    try {
      session.stopRecording();
      if(gtMode == GroundTruthMode.TRUE){
        groundTruthRecorder.recordBagStop();
      }
    } catch (RecordingFailedException e) {
      Log.e(TAG, "stopRecording - Failed to stop recording", e);
      return false;
    } catch (IOException e){

    }

    // Correctness checking: check if the session stopped recording.
    return session.getRecordingStatus() == RecordingStatus.NONE;
  }

  private void pauseARCoreSession() {
    // Pause the GLSurfaceView so that it doesn't update the ARCore session.
    // Pause the ARCore session so that we can update its configuration.
    // If the GLSurfaceView is not paused,
    //   onDrawFrame() will try to update the ARCore session
    //   while it's paused, resulting in a crash.
    resetMode();
    surfaceView.onPause();
    session.pause();
  }

  private boolean resumeARCoreSession() {
    // We must resume the ARCore session before the GLSurfaceView.
    // Otherwise, the GLSurfaceView will try to update the ARCore session.
    try {
      session.resume();
    } catch (CameraNotAvailableException e) {
      Log.e(TAG, "CameraNotAvailableException in resumeARCoreSession", e);
      return false;
    }

    surfaceView.onResume();
    return true;
  }

  private final String MP4_VIDEO_MIME_TYPE = "video/mp4";

  private String createMp4File() {
    String mp4FileName = recordType + Long.toString(scenario) + ".mp4";

    ContentResolver resolver = this.getContentResolver();

    Uri videoCollection = null;
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      videoCollection = MediaStore.Video.Media.getContentUri(
              MediaStore.VOLUME_EXTERNAL_PRIMARY);
    } else {
      videoCollection = MediaStore.Video.Media.EXTERNAL_CONTENT_URI;
    }

    // Create a new Media file record.
    ContentValues newMp4FileDetails = new ContentValues();
    newMp4FileDetails.put(MediaStore.Video.Media.DISPLAY_NAME, mp4FileName);
    newMp4FileDetails.put(MediaStore.Video.Media.MIME_TYPE, MP4_VIDEO_MIME_TYPE);

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      // The Relative_Path column is only available since API Level 29.
      newMp4FileDetails.put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES);
    } else {
      // Use the Data column to set path for API Level <= 28.
      File mp4FileDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
      String absoluteMp4FilePath = new File(mp4FileDir, mp4FileName).getAbsolutePath();
      newMp4FileDetails.put(MediaStore.Video.Media.DATA, absoluteMp4FilePath);
    }

    Uri newMp4FileUri = resolver.insert(videoCollection, newMp4FileDetails);

    // Ensure that this file exists and can be written.
    if (newMp4FileUri == null) {
      Log.e(TAG, String.format("Failed to insert Video entity in MediaStore. API Level = %d", Build.VERSION.SDK_INT));
      return null;
    }

    // This call ensures the file exist before we pass it to the ARCore API.
    if (!testFileWriteAccess(newMp4FileUri)) {
      return null;
    }

    String filePath = getMediaFilePath(newMp4FileUri);
    Log.d(TAG, String.format("createMp4File = %s, API Level = %d", filePath, Build.VERSION.SDK_INT));

    return filePath;
  }

  // Test if the file represented by the content Uri can be open with write access.
  private boolean testFileWriteAccess(Uri contentUri) {
    try (java.io.OutputStream mp4File = this.getContentResolver().openOutputStream(contentUri)) {
      Log.d(TAG, String.format("Success in testFileWriteAccess %s", contentUri.toString()));
      return true;
    } catch (java.io.FileNotFoundException e) {
      Log.e(TAG, String.format("FileNotFoundException in testFileWriteAccess %s", contentUri.toString()), e);
    } catch (java.io.IOException e) {
      Log.e(TAG, String.format("IOException in testFileWriteAccess %s", contentUri.toString()), e);
    }

    return false;
  }

  // Query the Media.DATA column to get file path from MediaStore content:// Uri
  private String getMediaFilePath(Uri mediaStoreUri) {
    String[] projection = { MediaStore.Images.Media.DATA };

    CursorLoader loader = new CursorLoader(this, mediaStoreUri, projection, null, null, null);
    Cursor cursor = loader.loadInBackground();
    cursor.moveToFirst();

    int data_column_index = cursor.getColumnIndexOrThrow(projection[0]);
    String data_result = cursor.getString(data_column_index);

    cursor.close();

    return data_result;
  }


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    setContentView(R.layout.activity_main);
    surfaceView = findViewById(R.id.surfaceview);
//    message_text = findViewById(R.id.messageText);
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    // Set up touch listener.
//    mainHandler = new MainHandler();
    tapHelper = new TapHelper(/*context=*/ this);
    surfaceView.setOnTouchListener(tapHelper);

    // Set up renderer.
    render = new SampleRender(surfaceView, this, getAssets());

    installRequested = false;

    depthSettings.onCreate(this);
    instantPlacementSettings.onCreate(this);

    gestureDetector =
            new GestureDetector(
                    this,
                    new GestureDetector.SimpleOnGestureListener() {
                      @Override
                      public boolean onSingleTapUp(MotionEvent e) {
                        synchronized (singleTapLock) {
                          if(recordType == RecordType.HOST){
                            touchCoordinate = new float[] {e.getX(), e.getY()};
                          }
                          if (currentMode == HostResolveMode.HOSTING) {
                            queuedSingleTap = e;
                          }
                        }
                        return true;
                      }

                      @Override
                      public boolean onDown(MotionEvent e) {
                        return true;
                      }
                    });
    surfaceView.setOnTouchListener((v, event) -> gestureDetector.onTouchEvent(event));

    // Initialize UI components.
    hostButton = findViewById(R.id.host_button);
    hostButton.setOnClickListener((view) -> onHostButtonPress());
    resolveButton = findViewById(R.id.resolve_button);
    resolveButton.setOnClickListener((view) -> onResolveButtonPress());
    scenarioButton = findViewById(R.id.scenario_button);
    scenarioButton.setOnClickListener((view) -> onScenarioButtonPress());
    imageHostButton = findViewById(R.id.image_hosting_button);
    imageHostButton.setOnClickListener((view) -> onImageHostButtonPress());
    roomCodeText = findViewById(R.id.room_code_text);

//    findViewById(R.id.playback_button).setVisibility(View.GONE);
    resolveButton.setVisibility(View.GONE);
//    hostButton.setVisibility(View.GONE);
//    imageHostButton.setVisibility(View.GONE);

    // Initialize Cloud Anchor variables.
    firebaseManager = new FirebaseManager(this);
    currentMode = HostResolveMode.NONE;
    sharedPreferences = getSharedPreferences(PREFERENCE_FILE_KEY, Context.MODE_PRIVATE);
    remoteServerManager = new RemoteServerManager(serverIP, serverPort);
    viewHandler = new ViewHandler();

    if(gtMode == GroundTruthMode.TRUE){
      groundTruthRecorder = new GroundTruthRecorder(this);
    }
  }


  class ViewHandler extends Handler{
    @Override
    public void handleMessage(@NonNull Message msg){
      super.handleMessage(msg);

      Bundle bundle = msg.getData();
      String value = bundle.getString("value");
      Log.d("yunho", value);
      if(value == "ImageHostButton"){
        onImageHostButtonPress();
      }
      if(value == "ResetMode"){
        resetMode();
      }
      if(value == "ResolveButton"){
        onResolveButtonPress();
      }
      if(value == "HostButton"){
        onHostButtonPress();
      }
//      if(value == "TouchEvent"){
//        float[] coordinates= bundle.getFloatArray("coordinate");
//        long downTime = SystemClock.uptimeMillis();
//        long eventTime = SystemClock.uptimeMillis() + 100;
//        MotionEvent motionEvent = MotionEvent.obtain(
//                downTime,
//                eventTime,
//                MotionEvent.ACTION_UP,
//                coordinates[0],
//                coordinates[1],
//                0
//        );
////        Log.d("yunho-event", motionEvent.toString());
//        gestureDetector.onTouchEvent(motionEvent);
////        surfaceView.dispatchTouchEvent(motionEvent);
//      }
    }
  }

  @Override
  protected void onDestroy() {
    if (session != null) {
      // Explicitly close ARCore Session to release native resources.
      // Review the API reference for important considerations before calling close() in apps with
      // more complicated lifecycle requirements:
      // https://developers.google.com/ar/reference/java/arcore/reference/com/google/ar/core/Session#close()
      session.close();
      session = null;
    }

    super.onDestroy();
  }

  @Override
  protected void onResume() {
    super.onResume();

    if (session == null) {
      Exception exception = null;
      String message = null;
      try {
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }

        // ARCore requires camera permissions to operate. If we did not yet obtain runtime
        // permission on Android M and above, now is a good time to ask the user for it.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }

        createSession();

      }catch (UnavailableDeviceNotCompatibleException e) {
        message = "This device does not support AR";
        exception = e;
      } catch (Exception e) {
        message = "Failed to create AR session";
        exception = e;
      }

      if (message != null) {
        messageSnackbarHelper.showError(this, message);
        Log.e(TAG, "Exception creating session", exception);
        return;
      }
    }

    // Note that order matters - see the note in onPause(), the reverse applies here.
    try {
      configureSession();
      // To record a live camera session for later playback, call
      // `session.startRecording(recorderConfig)` at anytime. To playback a previously recorded AR
      // session instead of using the live camera feed, call
      // `session.setPlaybackDataset(playbackDatasetPath)` before calling `session.resume()`. To
      // learn more about recording and playback, see:
      // https://developers.google.com/ar/develop/java/recording-and-playback
      session.resume();
    } catch (CameraNotAvailableException e) {
      messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
      session = null;
      return;
    }

    surfaceView.onResume();
    displayRotationHelper.onResume();
  }

  private void createSession() {
    if (session == null) {
      Exception exception = null;
      int messageId = -1;
      try {
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }
        // ARCore requires camera permissions to operate. If we did not yet obtain runtime
        // permission on Android M and above, now is a good time to ask the user for it.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }
        session = new Session(this);
      } catch (UnavailableArcoreNotInstalledException e) {
        messageId = R.string.snackbar_arcore_unavailable;
        exception = e;
      } catch (UnavailableApkTooOldException e) {
        messageId = R.string.snackbar_arcore_too_old;
        exception = e;
      } catch (UnavailableSdkTooOldException e) {
        messageId = R.string.snackbar_arcore_sdk_too_old;
        exception = e;
      } catch (Exception e) {
        messageId = R.string.snackbar_arcore_exception;
        exception = e;
      }

      if (exception != null) {
        snackbarHelper.showError(this, getString(messageId));
        Log.e(TAG, "Exception creating session", exception);
        return;
      }

      // Create default config and check if supported.
      Config config = new Config(session);
      config.setCloudAnchorMode(Config.CloudAnchorMode.ENABLED);
      session.configure(config);

      // Setting the session in the HostManager.
      cloudManager.setSession(session);
    }

    // Note that order matters - see the note in onPause(), the reverse applies here.
    try {
      session.resume();
    } catch (CameraNotAvailableException e) {
      snackbarHelper.showError(this, getString(R.string.snackbar_camera_unavailable));
      session = null;
      return;
    }
  }

  private void createSessionNoResume(){
    session.close();
    try {
      session = new Session(this);
    } catch (UnavailableArcoreNotInstalledException
            |UnavailableApkTooOldException
            |UnavailableSdkTooOldException
            |UnavailableDeviceNotCompatibleException e) {
      Log.e(TAG, "Error in return to Idle state. Cannot create new ARCore session", e);
    }
    configureSession();
  }

  @Override
  public void onPause() {
    super.onPause();
    if (session != null) {
      // Note that the order matters - GLSurfaceView is paused first so that it does not try
      // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
      // still call session.update() and get a SessionPausedException.
      displayRotationHelper.onPause();
      surfaceView.onPause();
      session.pause();
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
    super.onRequestPermissionsResult(requestCode, permissions, results);
    if (android.os.Build.VERSION.SDK_INT > android.os.Build.VERSION_CODES.O &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSIONS_REQUEST_CAMERA);
      return;
    }

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
      return;
    }

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSIONS_REQUEST_WRITE);
      return;
    }

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
      return;
    }

    if (!CameraPermissionHelper.hasCameraPermission(this)) {
      // Use toast instead of snackbar here since the activity will exit.
      Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
          .show();
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this);
      }
      finish();
    }
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
  }

  @Override
  public void onSurfaceCreated(SampleRender render) {
    // Prepare the rendering objects. This involves reading shaders and 3D model files, so may throw
    // an IOException.
    try {
      planeRenderer = new PlaneRenderer(render);
      textRenderer = new TextRenderer(render, PLANE_TEXT_WIDTH, PLANE_TEXT_HEIGHT);
      planeanchorRenderer = new PlaneanchorRenderer(render);
      backgroundRenderer = new BackgroundRenderer(render);
      virtualSceneFramebuffer = new Framebuffer(render, /*width=*/ 1, /*height=*/ 1);

      cubemapFilter =
          new SpecularCubemapFilter(
              render, CUBEMAP_RESOLUTION, CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES);
      // Load DFG lookup table for environmental lighting
      dfgTexture =
          new Texture(
              render,
              Texture.Target.TEXTURE_2D,
              Texture.WrapMode.CLAMP_TO_EDGE,
              /*useMipmaps=*/ false);
      // The dfg.raw file is a raw half-float texture with two channels.
      final int dfgResolution = 64;
      final int dfgChannels = 2;
      final int halfFloatSize = 2;

      ByteBuffer buffer =
          ByteBuffer.allocateDirect(dfgResolution * dfgResolution * dfgChannels * halfFloatSize);
      try (InputStream is = getAssets().open("models/dfg.raw")) {
        is.read(buffer.array());
      }
      // SampleRender abstraction leaks here.
      GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dfgTexture.getTextureId());
      GLError.maybeThrowGLException("Failed to bind DFG texture", "glBindTexture");
      GLES30.glTexImage2D(
          GLES30.GL_TEXTURE_2D,
          /*level=*/ 0,
          GLES30.GL_RG16F,
          /*width=*/ dfgResolution,
          /*height=*/ dfgResolution,
          /*border=*/ 0,
          GLES30.GL_RG,
          GLES30.GL_HALF_FLOAT,
          buffer);
      GLError.maybeThrowGLException("Failed to populate DFG texture", "glTexImage2D");

      // Point cloud
      pointCloudShader =
          Shader.createFromAssets(
                  render, "shaders/point_cloud.vert", "shaders/point_cloud.frag", /*defines=*/ null)
              .setVec4(
                  "u_Color", new float[] {31.0f / 255.0f, 188.0f / 255.0f, 210.0f / 255.0f, 1.0f})
              .setFloat("u_PointSize", 5.0f);
      // four entries per vertex: X, Y, Z, confidence
      pointCloudVertexBuffer =
          new VertexBuffer(render, /*numberOfEntriesPerVertex=*/ 4, /*entries=*/ null);
      final VertexBuffer[] pointCloudVertexBuffers = {pointCloudVertexBuffer};
      pointCloudMesh =
          new Mesh(
              render, Mesh.PrimitiveMode.POINTS, /*indexBuffer=*/ null, pointCloudVertexBuffers);

      // Virtual object to render (ARCore pawn)
      Texture virtualObjectAlbedoTexture =
          Texture.createFromAsset(
              render,
              "models/pawn_albedo.png",
              Texture.WrapMode.CLAMP_TO_EDGE,
              Texture.ColorFormat.SRGB);
      Texture virtualObjectPbrTexture =
          Texture.createFromAsset(
              render,
              "models/pawn_roughness_metallic_ao.png",
              Texture.WrapMode.CLAMP_TO_EDGE,
              Texture.ColorFormat.LINEAR);
      virtualObjectMesh = Mesh.createFromAsset(render, "models/pawn.obj");
      virtualObjectShader =
          Shader.createFromAssets(
                  render,
                  "shaders/environmental_hdr.vert",
                  "shaders/environmental_hdr.frag",
                  /*defines=*/ new HashMap<String, String>() {
                    {
                      put(
                          "NUMBER_OF_MIPMAP_LEVELS",
                          Integer.toString(cubemapFilter.getNumberOfMipmapLevels()));
                    }
                  })
              .setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture)
              .setTexture("u_RoughnessMetallicAmbientOcclusionTexture", virtualObjectPbrTexture)
              .setTexture("u_Cubemap", cubemapFilter.getFilteredCubemapTexture())
              .setTexture("u_DfgTexture", dfgTexture);

      textureA = Texture.createFromAsset(
              render, "models/google.png", Texture.WrapMode.REPEAT, Texture.ColorFormat.LINEAR);
      textureB = Texture.createFromAsset(
              render, "models/google.png", Texture.WrapMode.REPEAT, Texture.ColorFormat.LINEAR);

      Square planeSquare = new Square(50, 50);
      IndexBuffer planeIndexBuffer = new IndexBuffer(render, planeSquare.getIndexBuffer());
      VertexBuffer[] planeVertexBuffers = {
              new VertexBuffer(render, 3, /*entries=*/ planeSquare.getVertexBuffer()),
              new VertexBuffer(render, 2, planeSquare.getTextureCoordinateBuffer())};
      anchorMesh = new Mesh(render, Mesh.PrimitiveMode.TRIANGLES, planeIndexBuffer, planeVertexBuffers);

      cloudAnchorTextRenderer = new TextRenderer(render, 50, 50);

      Square cloudAnchorSquare = new Square(50, 50);
      IndexBuffer cloudAnchorIndexBuffer = new IndexBuffer(render, cloudAnchorSquare.getIndexBuffer());
      VertexBuffer[] cloudAnchorVertexBuffers = {
              new VertexBuffer(render, 3, /*entries=*/ cloudAnchorSquare.getVertexBuffer()),
              new VertexBuffer(render, 2, cloudAnchorSquare.getTextureCoordinateBuffer())};
      planeMesh = new Mesh(render, Mesh.PrimitiveMode.TRIANGLE_STRIP, cloudAnchorIndexBuffer, cloudAnchorVertexBuffers);
      Bitmap bitmap = textBitmap(this, 50, 50, "Cloud Anchor", 14);
      cloudAnchorTextTexture = Texture.createFromBitmap(
              render, bitmap, Texture.WrapMode.REPEAT, Texture.ColorFormat.LINEAR);

      Square mSquare = new Square(6000, 6000);
      IndexBuffer indexBufferObject = new IndexBuffer(render, /*entries=*/ mSquare.getIndexBuffer());
      VertexBuffer[] vertexBuffersB = {
              new VertexBuffer(render, 3, /*entries=*/ mSquare.getVertexBuffer()),
              new VertexBuffer(render, 2, mSquare.getTextureCoordinateBuffer())};
      planeMeshA = new Mesh(render, Mesh.PrimitiveMode.TRIANGLES, indexBufferObject, vertexBuffersB);

    } catch (IOException e) {
      Log.e(TAG, "Failed to read a required asset file", e);
      messageSnackbarHelper.showError(this, "Failed to read a required asset file: " + e);
    }
  }

  @Override
  public void onSurfaceChanged(SampleRender render, int width, int height) {
    displayRotationHelper.onSurfaceChanged(width, height);
    virtualSceneFramebuffer.resize(width, height);
  }

  @Override
  public void onDrawFrame(SampleRender render) {
    if (session == null) {
      return;
    }

    if (appState == AppState.Playingback
            && session.getPlaybackStatus() == PlaybackStatus.FINISHED) {
      this.runOnUiThread(this::stopPlayingback);
      session = null;
      createSession();
      Message message = viewHandler.obtainMessage();
      Bundle bundle = new Bundle();
      bundle.putString("value", "ResetMode");
      message.setData(bundle);
      viewHandler.sendMessage(message);
      return;
    }
    if (appState == AppState.Playingback || appState == AppState.Recording){
      counter += 1;
      if (resolveCalled == false){
        if (counter > 10){
          Message message = viewHandler.obtainMessage();
          Bundle bundle = new Bundle();
          bundle.putString("value", "HostButton");
          message.setData(bundle);
          viewHandler.sendMessage(message);
          resolveCalled = true;
        }
      }
    }

    // Texture names should only be set once on a GL thread unless they change. This is done during
    // onDrawFrame rather than onSurfaceCreated since the session is not guaranteed to have been
    // initialized during the execution of onSurfaceCreated.
    if (!hasSetTextureNames) {
      session.setCameraTextureNames(
          new int[] {backgroundRenderer.getCameraColorTexture().getTextureId()});
      hasSetTextureNames = true;
    }

    // -- Update per-frame state

    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session);

    // Obtain the current frame from ARSession. When the configuration is set to
    // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
    // camera framerate.
    Frame frame;
    try {
      frame = session.update();
    } catch (CameraNotAvailableException e) {
      Log.e(TAG, "Camera not available during onDrawFrame", e);
      messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
      return;
    }
    Camera camera = frame.getCamera();

    if (appState == AppState.Recording){
      try{
        if(recordType == RecordType.HOST && imageHostingTouched == true){
          ByteBuffer touchData = ByteBuffer.wrap(floatArrayToByteArray(imageHostingCoordinate));
          Log.d("yunho-imagehosting", Arrays.toString(imageHostingCoordinate));
          frame.recordTrackData(trackUUID, touchData);
          imageHostingTouched = false;
        }
        if(recordType == RecordType.HOST && doneTouched == true){
          ByteBuffer touchData = ByteBuffer.wrap(floatArrayToByteArray(imageHostingCoordinate));
          Log.d("yunho-done", Arrays.toString(imageHostingCoordinate));
          frame.recordTrackData(trackUUID, touchData);
          doneTouched = false;
        }
      }catch(IOException e){
        Log.e(TAG, "record touch event failed", e);
      }
    }

    if (appState == AppState.Playingback){
      Collection<TrackData> trackDataList = frame.getUpdatedTrackData(trackUUID);
      for (TrackData trackData : trackDataList){
        ByteBuffer bytes = trackData.getData();
        touchCoordinate = byteBufferToFloatArray(bytes);
        Log.d("yunho-coordinate", Arrays.toString(touchCoordinate));
        if(Arrays.equals(touchCoordinate, imageHostingCoordinate)){
          Message message = viewHandler.obtainMessage();
          Bundle bundle = new Bundle();
          bundle.putString("value", "ImageHostButton");
          message.setData(bundle);
          viewHandler.sendMessage(message);
        }
        else{
          long downTime = SystemClock.uptimeMillis();
          long eventTime = SystemClock.uptimeMillis() + 100;
          MotionEvent motionEvent = MotionEvent.obtain(
                  downTime,
                  eventTime,
                  MotionEvent.ACTION_UP,
                  touchCoordinate[0],
                  touchCoordinate[1],
                  0
          );
          queuedSingleTap = motionEvent;
        }
      }
    }

    // Update BackgroundRenderer state to match the depth settings.
    try {
      backgroundRenderer.setUseDepthVisualization(
          render, depthSettings.depthColorVisualizationEnabled());
      backgroundRenderer.setUseOcclusion(render, depthSettings.useDepthForOcclusion());
    } catch (IOException e) {
      Log.e(TAG, "Failed to read a required asset file", e);
      messageSnackbarHelper.showError(this, "Failed to read a required asset file: " + e);
      return;
    }
    // BackgroundRenderer.updateDisplayGeometry must be called every frame to update the coordinates
    // used to draw the background camera image.
    backgroundRenderer.updateDisplayGeometry(frame);

    if (camera.getTrackingState() == TrackingState.TRACKING
        && (depthSettings.useDepthForOcclusion()
            || depthSettings.depthColorVisualizationEnabled())) {
      try (Image depthImage = frame.acquireDepthImage()) {
        backgroundRenderer.updateCameraDepthTexture(depthImage);
      } catch (NotYetAvailableException e) {
        // This normally means that depth data is not available yet. This is normal so we will not
        // spam the logcat with this.
      }
    }

    TrackingState cameraTrackingState = camera.getTrackingState();

    cloudManager.onUpdate();

    // Handle one tap per frame.
    handleTap(frame, cameraTrackingState);

    // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
    trackingStateHelper.updateKeepScreenOnFlag(camera.getTrackingState());

    // Show a message based on whether tracking has failed, if planes are detected, and if the user
    // has placed any objects.
    String message = null;
    if (currentMode == HostResolveMode.PLANEHOSTING){
      message = HOSTING_PLANE_MESSAGE;
    } else if (camera.getTrackingState() == TrackingState.PAUSED) {
      if (camera.getTrackingFailureReason() == TrackingFailureReason.NONE) {
        message = SEARCHING_PLANE_MESSAGE;
      } else {
        message = TrackingStateHelper.getTrackingFailureReasonString(camera);
      }
    } else if (hasTrackingPlane()) {
      if (anchors.isEmpty()) {
        message = WAITING_FOR_TAP_MESSAGE;
      }
    } else {
      message = SEARCHING_PLANE_MESSAGE;
    }
    if (message == null) {
      messageSnackbarHelper.hide(this);
    } else {
      messageSnackbarHelper.showMessage(this, message);
    }

    // -- Draw background

    if (frame.getTimestamp() != 0) {
      // Suppress rendering if the camera did not produce the first frame yet. This is to avoid
      // drawing possible leftover data from previous sessions if the texture is reused.
      backgroundRenderer.drawBackground(render);
    }

    // If not tracking, don't draw 3D objects.
    if (camera.getTrackingState() == TrackingState.PAUSED) {
      return;
    }

    // -- Draw non-occluded virtual objects (planes, point cloud)

    // Get projection matrix.
    camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR);

    // Get camera matrix and draw.
    camera.getViewMatrix(viewMatrix, 0);

    // Visualize tracked points.
    // Use try-with-resources to automatically release the point cloud.
    try (PointCloud pointCloud = frame.acquirePointCloud()) {
      if (pointCloud.getTimestamp() > lastPointCloudTimestamp) {
        Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0);
        pointCloudShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);

        pointCloudVertexBuffer.set(pointCloud.getPoints());
        lastPointCloudTimestamp = pointCloud.getTimestamp();
        if (currentMode == HostResolveMode.PLANEHOSTING){
//          Log.d("yunho-id", Integer.toString(pointCloud.getIds().capacity()));

          if(pointCloud.getIds().capacity() > keyFrameThreshold){
            int[] currentPointIds = new int[pointCloud.getIds().remaining()];
            pointCloud.getIds().get(currentPointIds);
            Arrays.asList(currentPointIds);
            Set<Integer> currentPointSet = Arrays.stream(currentPointIds).boxed().collect(Collectors.toSet());
//            keyFrameSelector
//            using point intersection between previous keyframe and currentframe
            if(CountIntersection(previousKeyFrameIds, currentPointSet) <= 0){
              Log.d("yunho", "keyFrameSelected "+Integer.toString(keyFrameCount));
              Log.d("yunho-pcd_previous", previousKeyFrameIds.toString());
              Log.d("yunho-pcd_current", currentPointSet.toString());
              previousKeyFrameIds = currentPointSet;
              keyFrameCount += 1;
              try{
                Image keyFrame = frame.acquireCameraImage();
                byte[] jpegData = ImageHelper.imageToByteArray(keyFrame);
//                Matrix.multiplyMM(ViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0);
                byte[] transformed_pcd = transformPointCloud(pointCloud.getPoints(), viewMatrix);
//                byte[] pcd = pointCloudToByteArray(transformed_pcd);
                Log.d("yunho", "keyFrameSelected "+Integer.toString(keyFrameCount));
                anchor.getPose().toMatrix(modelMatrix, 0);
                Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0);
                Matrix.invertM(invModelViewMatrix, 0, modelViewMatrix, 0);
                firebaseManager.uploadImage(jpegData, String.format("%03d", keyFrameCount), invModelViewMatrix, Long.toString(frame.getAndroidCameraTimestamp()));
                firebaseManager.uploadPCD(transformed_pcd, String.format("%03d", keyFrameCount));
                keyFrame.close();
              }
              catch(Exception e){
                e.printStackTrace();
              }
            }
          }
        }

      }
      render.draw(pointCloudMesh, pointCloudShader);
    }

    if(currentMode == HostResolveMode.PLANEHOSTING){
      planeRenderer.drawPlanes(
              render,
              session.getAllTrackables(Plane.class),
              camera.getDisplayOrientedPose(),
              projectionMatrix);
    }

    PlaneAnchor uploadedAnchor = null;
    if(STORE_ARCORE_PLANES == 1){
      firebaseManager.cleanServerPlanes(method);
      int i = 1;
      for(Plane plane : session.getAllTrackables(Plane.class)){
        if (plane.getTrackingState() != TrackingState.TRACKING || plane.getSubsumedBy() != null) {
          continue;
        }

        float distance = PlaneRenderer.calculateDistanceToPlane(plane.getCenterPose(), camera.getDisplayOrientedPose());
        if (distance < 0) { // Plane is back-facing.
          continue;
        }
        uploadedAnchor = firebaseManager.uploadPlane(plane, anchor, keyFrameCount, method);
        if(uploadedAnchor != null){
          makePlaneRenders(uploadedAnchor, "PLANE"+Integer.toString(i));
          i+=1;
        }
      }
      firebaseManager.cleanPlaneID();
      STORE_ARCORE_PLANES = 0;
//      setNewAnchor(null);
    }


    // -- Draw occluded virtual objects

    // Update lighting parameters in the shader
    updateLightEstimation(frame.getLightEstimate(), viewMatrix);
    // Visualize anchors created by touch.
    render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f);

    boolean shouldDrawAnchor = false;
    synchronized (anchorLock) {
      if (anchor != null && anchor.getTrackingState() == TrackingState.TRACKING) {
        // Get the current pose of an Anchor in world space. The Anchor pose is updated
        // during calls to session.update() as ARCore refines its estimate of the world.
        shouldDrawAnchor = true;
      }
    }

    if(shouldDrawAnchor==true) {
      // Get the current pose of an Anchor in world space. The Anchor pose is updated
      // during calls to session.update() as ARCore refines its estimate of the world.
      anchor.getPose().toMatrix(modelMatrix, 0);

      // Calculate model/view/projection matrices
      Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0);
      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0);

      // Update shader properties and draw
      virtualObjectShader.setMat4("u_ModelView", modelViewMatrix);
      virtualObjectShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);

      planeanchorRenderer.drawPlaneAnchors(
              render,
              modelMatrix,
              camera.getDisplayOrientedPose(),
              projectionMatrix,
              anchorMesh,
              textureA,
              anchorColor,
              anchorColor
      );

//      cloudAnchorTextRenderer.drawPlaneText(
//              render,
//              modelMatrix,
//              camera.getDisplayOrientedPose(),
//              projectionMatrix,
//              cloudAnchorTextTexture
//      );

      if(shouldMakeRenders==true){
        planeRenders.clear();
        int i = 0;
        for(PlaneAnchor planeAnchor: planeAnchors){
          makePlaneRenders(planeAnchor, "PLANE"+Integer.toString(i));
          i+=1;
        }
        shouldMakeRenders=false;
      }

      int i = 0;
      int[] colorBIndex = {};
      float[] color;
      float[] borderColor;
      for(PlaneRender planeRender: planeRenders){
        if(Arrays.binarySearch(colorBIndex, i) >= 0){
          color = planeAnchorColorB;
          borderColor= borderColorB;
        }
        else{
          color = planeAnchorColor;
          borderColor= borderColorA;
        }
        float[] planeAnchorMatrix = new float[16];
        Matrix.multiplyMM(planeAnchorMatrix, 0, modelMatrix, 0, planeRender.getPlaneMatrix(), 0);
        planeanchorRenderer.drawPlaneAnchors(
                render,
                planeAnchorMatrix,
                camera.getDisplayOrientedPose(),
                projectionMatrix,
                planeRender.getPlaneMesh(),
                textureA,
                color,
                borderColor
        );

        textRenderer.drawPlaneText(
                render,
                planeAnchorMatrix,
                camera.getDisplayOrientedPose(),
                projectionMatrix,
                planeRender.getPlaneTexture()
        );
        i += 1;
      }
    }

    // Compose the virtual scene with the background.
    backgroundRenderer.drawVirtualScene(render, virtualSceneFramebuffer, Z_NEAR, Z_FAR);
  }

  public static int CountIntersection(Set<Integer> set1, Set<Integer> set2) {
    Set<Integer> a;
    Set<Integer> b;
    if (set1.size() <= set2.size()) {
      a = set1;
      b = set2;
    } else {
      a = set2;
      b = set1;
    }
    int count = 0;
    for (Integer e : a) {
      if (b.contains(e)) {
        count++;
      }
    }
    return count;
  }

  // Handle only one tap per frame, as taps are usually low frequency compared to frame rate.
  private void handleTap(Frame frame, TrackingState cameraTrackingState) {
    // Handle taps. Handling only one tap per frame, as taps are usually low frequency
    // compared to frame rate.
    synchronized (singleTapLock) {
      synchronized (anchorLock) {
        // Only handle a tap if the anchor is currently null, the queued tap is non-null and the
        // camera is currently tracking.
        if (anchor == null
                && queuedSingleTap != null
                && cameraTrackingState == TrackingState.TRACKING) {
          Preconditions.checkState(
                  currentMode == HostResolveMode.HOSTING,
                  "We should only be creating an anchor in hosting mode.");
          for (HitResult hit : frame.hitTest(queuedSingleTap)) {
            if (shouldCreateAnchorWithHit(hit)) {
              try{
                if(recordType == RecordType.HOST && anchorTouched == false){
                  ByteBuffer touchData = ByteBuffer.wrap(floatArrayToByteArray(touchCoordinate));
                  Log.d("yunho-anchor", Arrays.toString(touchCoordinate));
                  frame.recordTrackData(trackUUID, touchData);
                  anchorTouched = true;
                }
              }catch(IOException e){
                Log.e(TAG, "anchor record failed",e);
              }
              Anchor newAnchor = hit.createAnchor();
              Preconditions.checkNotNull(hostListener, "The host listener cannot be null.");
              cloudManager.hostCloudAnchor(newAnchor, hostListener);
              setNewAnchor(newAnchor);
              snackbarHelper.showMessage(this, getString(R.string.snackbar_anchor_placed));
              break; // Only handle the first valid hit.
            }
          }
        }
      }
      queuedSingleTap = null;
    }
  }

  private static boolean shouldCreateAnchorWithHit(HitResult hit) {
    Trackable trackable = hit.getTrackable();
    if (trackable instanceof Plane) {
      // Check if the hit was within the plane's polygon.
      return ((Plane) trackable).isPoseInPolygon(hit.getHitPose());
    } else if (trackable instanceof Point) {
      // Check if the hit was against an oriented point.
      return ((Point) trackable).getOrientationMode() == OrientationMode.ESTIMATED_SURFACE_NORMAL;
    }
    return false;
  }

  private void setNewAnchor(Anchor newAnchor) {
    synchronized (anchorLock) {
      if (anchor != null) {
        anchor.detach();
      }
      anchor = newAnchor;
    }
  }

  /** Callback function invoked when the Host Button is pressed. */
  private void onHostButtonPress() {
    if (currentMode == HostResolveMode.HOSTING || currentMode == HostResolveMode.HOSTINGDONE) {
      resetMode();
      return;
    }
    if(appState == AppState.Recording || appState == AppState.Playingback){
      onPrivacyAcceptedForHost();
    }
    else{
      if (!sharedPreferences.getBoolean(ALLOW_SHARE_IMAGES_KEY, false)) {
        showNoticeDialog(this::onPrivacyAcceptedForHost);
      } else {
        onPrivacyAcceptedForHost();
      }
    }
  }

  private void onImageHostButtonPress() {
    if (currentMode == HostResolveMode.HOSTINGDONE) {
      imageHostingTouched=true;
      imageHostButton.setText(R.string.hosting_done);
      firebaseManager.cleanServerImages();
      currentMode = HostResolveMode.PLANEHOSTING;
//      snackbarHelper.showMessageWithDismiss(this, getString(R.string.snackbar_on_host_image));
      return;
    }
    else if (currentMode == HostResolveMode.PLANEHOSTING){
      doneTouched=true;
//      snackbarHelper.showMessageWithDismiss(this, getString(R.string.snackbar_on_host_image_done));
      STORE_ARCORE_PLANES = 1;
//      if(recentRoomCode != 0L){
//        remoteServerManager.sendRoomCode(recentRoomCode);
//      }
      myResetMode();
      return;
    }
  }

  private void onPrivacyAcceptedForHost() {
    if (hostListener != null) {
      return;
    }
    resolveButton.setEnabled(false);
    hostButton.setText(R.string.cancel);
    imageHostButton.setVisibility(View.VISIBLE);
    snackbarHelper.showMessageWithDismiss(this, getString(R.string.snackbar_on_host));

    hostListener = new RoomCodeAndCloudAnchorIdListener();
    hostListener.onNewRoomCode(scenario);
//    firebaseManager.getNewRoomCode(hostListener);
  }

  /** Callback function invoked when the Resolve Button is pressed. */
  private void onResolveButtonPress() {
    if (currentMode == HostResolveMode.RESOLVING) {
      resetMode();
      return;
    }
    if (!sharedPreferences.getBoolean(ALLOW_SHARE_IMAGES_KEY, false)) {
      showNoticeDialog(this::onPrivacyAcceptedForResolve);
    } else {
      onPrivacyAcceptedForResolve();
    }
  }

  private void onPrivacyAcceptedForResolve() {
    ResolveDialogFragment dialogFragment = new ResolveDialogFragment();
    dialogFragment.setOkListener(this::onRoomCodeEntered);
    dialogFragment.show(getSupportFragmentManager(), "ResolveDialog");
  }

  private void onScenarioButtonPress(){
    ResolveDialogFragment dialogFragment = new ResolveDialogFragment();
    dialogFragment.setOkListener(this::onScenarioEntered);
    dialogFragment.show(getSupportFragmentManager(), "ScenarioDialog");
  }

  private void onScenarioEntered(Long roomCode) {
    roomCodeText.setText(String.valueOf(roomCode));
    scenario = roomCode;
  }

  /** Resets the mode of the app to its initial state and removes the anchors. */
  private void resetMode() {
    hostButton.setText(R.string.host_button_text);
    hostButton.setEnabled(true);
    resolveButton.setText(R.string.resolve_button_text);
//    resolveButton.setEnabled(true);
    scenarioButton.setEnabled(true);
    imageHostButton.setText(R.string.host_image_button_text);
    imageHostButton.setVisibility(View.GONE);
//    roomCodeText.setText(R.string.initial_room_code);
    currentMode = HostResolveMode.NONE;
    firebaseManager.clearRoomListener();
    hostListener = null;
    setNewAnchor(null);
    snackbarHelper.hide(this);
    cloudManager.clearListeners();
    keyFrameCount = 0;
    planeRenders.clear();
    planeAnchors.clear();
    resolveCalled = false;
    anchorTouched = false;
    imageHostingTouched = false;
    doneTouched = false;
    counter = 0;
  }

  private void myResetMode() {
    hostButton.setText(R.string.host_button_text);
    hostButton.setEnabled(true);
    resolveButton.setText(R.string.resolve_button_text);
//    resolveButton.setEnabled(true);
    scenarioButton.setEnabled(true);
    imageHostButton.setText(R.string.host_image_button_text);
    imageHostButton.setVisibility(View.GONE);
//    roomCodeText.setText(R.string.initial_room_code);
    currentMode = HostResolveMode.NONE;
    firebaseManager.clearRoomListener();
    hostListener = null;
    snackbarHelper.hide(this);
    cloudManager.clearListeners();
//    resolveCalled = false;
//    anchorTouched = false;
//    imageHostingTouched = false;
//    doneTouched = false;
  }

  /** Callback function invoked when the user presses the OK button in the Resolve Dialog. */
  private void onRoomCodeEntered(Long roomCode) {
    currentMode = HostResolveMode.RESOLVING;
    hostButton.setEnabled(false);
    resolveButton.setText(R.string.cancel);
    roomCodeText.setText(String.valueOf(roomCode));
//    snackbarHelper.showMessageWithDismiss(this, getString(R.string.snackbar_on_resolve));

    // Register a new listener for the given room.
    firebaseManager.registerNewListenerForRoom(
            roomCode,
            cloudAnchorId -> {
              // When the cloud anchor ID is available from Firebase.
              CloudAnchorResolveStateListener resolveListener =
                      new CloudAnchorResolveStateListener(roomCode);
              Preconditions.checkNotNull(resolveListener, "The resolve listener cannot be null.");
              cloudManager.resolveCloudAnchor(
                      cloudAnchorId, resolveListener, SystemClock.uptimeMillis());
            });

    firebaseManager.readData(firebaseManager.getPlaneAnchorsRef(roomCode, method), new myCallBack() {
      @Override
      public void onSuccess(ArrayList<PlaneAnchor> result) {
        planeAnchors.addAll(result);
        shouldMakeRenders=true;
      }
    });
  }

  public interface myCallBack{
    void onSuccess(ArrayList<PlaneAnchor> result);
  }
  private Mesh planeMesh;

  public void makePlaneRenders(PlaneAnchor planeAnchor, String name) {
    Square planeSquare = new Square(planeAnchor.getWidth(), planeAnchor.getHeight());
    IndexBuffer planeIndexBuffer = new IndexBuffer(render, planeSquare.getIndexBuffer());
    VertexBuffer[] planeVertexBuffers = {
            new VertexBuffer(render, 3, /*entries=*/ planeSquare.getVertexBuffer()),
            new VertexBuffer(render, 2, planeSquare.getTextureCoordinateBuffer())};
    planeMesh = new Mesh(render, Mesh.PrimitiveMode.TRIANGLE_STRIP, planeIndexBuffer, planeVertexBuffers);
    Bitmap bitmap = textBitmap(this, PLANE_TEXT_WIDTH, PLANE_TEXT_HEIGHT, name, 48);
    try{
      Texture planeName = Texture.createFromBitmap(
              render, bitmap, Texture.WrapMode.REPEAT, Texture.ColorFormat.LINEAR);
      planeRenders.add(new PlaneRender(planeMesh, planeAnchor.getTransformationMatrix(), planeName));
    }catch(IOException e){

    }
  }

  public Bitmap textBitmap(Context context, int width, int height, String str, int textSize){
    Paint paint = new Paint();
    paint.setTextSize(textSize);
    paint.setAntiAlias(true);
    paint.setARGB(0Xff, 0xFF, 0x00, 0x00);
    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    Canvas canvas = new Canvas(bitmap);
    bitmap.eraseColor(0xFFFFFFFF);

    int ascent = (int) Math.ceil(-paint.ascent());
    int descent = (int) Math.ceil(paint.descent());
    int measuredTextWidth = (int) Math.ceil(paint.measureText(str));

    int textHeight = ascent + descent;
    int textWidth = measuredTextWidth;

    canvas.drawText(str, (width - textWidth)/2f, (height + ascent + descent) / 2f, paint);

    return bitmap;
  }

  /**
   * Listens for both a new room code and an anchor ID, and shares the anchor ID in Firebase with
   * the room code when both are available.
   */
  private final class RoomCodeAndCloudAnchorIdListener
          implements CloudAnchorManager.CloudAnchorHostListener, FirebaseManager.RoomCodeListener {

    private Long roomCode;
    private String cloudAnchorId;

    @Override
    public void onNewRoomCode(Long newRoomCode) {
      Preconditions.checkState(roomCode == null, "The room code cannot have been set before.");
      roomCode = newRoomCode;
      roomCodeText.setText(String.valueOf(roomCode));
//      snackbarHelper.showMessageWithDismiss(
//              HelloArActivity.this, getString(R.string.snackbar_room_code_available));
      checkAndMaybeShare();
      synchronized (singleTapLock) {
        // Change currentMode to HOSTING after receiving the room code (not when the 'Host' button
        // is tapped), to prevent an anchor being placed before we know the room code and able to
        // share the anchor ID.
        currentMode = HostResolveMode.HOSTING;
      }
    }

    @Override
    public void onError(DatabaseError error) {
      Log.w(TAG, "A Firebase database error happened.", error.toException());
      snackbarHelper.showError(
              HelloArActivity.this, getString(R.string.snackbar_firebase_error));
    }

    @Override
    public void onCloudTaskComplete(Anchor anchor) {
      Anchor.CloudAnchorState cloudState = anchor.getCloudAnchorState();
      if (cloudState.isError()) {
        Log.e(TAG, "Error hosting a cloud anchor, state " + cloudState);
        snackbarHelper.showMessageWithDismiss(
                HelloArActivity.this, getString(R.string.snackbar_host_error, cloudState));
        return;
      }
      Preconditions.checkState(
              cloudAnchorId == null, "The cloud anchor ID cannot have been set before.");
      cloudAnchorId = anchor.getCloudAnchorId();
      setNewAnchor(anchor);
      checkAndMaybeShare();
    }

    private void checkAndMaybeShare() {
      if (roomCode == null || cloudAnchorId == null) {
        return;
      }
      firebaseManager.storeAnchorIdInRoom(roomCode, cloudAnchorId);
      currentMode = HostResolveMode.HOSTINGDONE;
      snackbarHelper.showMessageWithDismiss(
              HelloArActivity.this, getString(R.string.snackbar_cloud_id_shared));
      recentRoomCode = roomCode;
    }
  }

  private final class CloudAnchorResolveStateListener
          implements CloudAnchorManager.CloudAnchorResolveListener {
    private final long roomCode;

    CloudAnchorResolveStateListener(long roomCode) {
      this.roomCode = roomCode;
    }

    @Override
    public void onCloudTaskComplete(Anchor anchor) {
      // When the anchor has been resolved, or had a final error state.
      Anchor.CloudAnchorState cloudState = anchor.getCloudAnchorState();
      if (cloudState.isError()) {
        Log.w(
                TAG,
                "The anchor in room "
                        + roomCode
                        + " could not be resolved. The error state was "
                        + cloudState);
        snackbarHelper.showMessageWithDismiss(
                HelloArActivity.this, getString(R.string.snackbar_resolve_error, cloudState));
        return;
      }
      snackbarHelper.showMessageWithDismiss(
              HelloArActivity.this, getString(R.string.snackbar_resolve_success));
      setNewAnchor(anchor);
    }

    @Override
    public void onShowResolveMessage() {
      snackbarHelper.setMaxLines(4);
      snackbarHelper.showMessageWithDismiss(
              HelloArActivity.this, getString(R.string.snackbar_resolve_no_result_yet));
    }
  }

  public void showNoticeDialog(HostResolveListener listener) {
    DialogFragment dialog = PrivacyNoticeDialogFragment.createDialog(listener);
    dialog.show(getSupportFragmentManager(), PrivacyNoticeDialogFragment.class.getName());
  }

  @Override
  public void onDialogPositiveClick(DialogFragment dialog) {
    if (!sharedPreferences.edit().putBoolean(ALLOW_SHARE_IMAGES_KEY, true).commit()) {
      throw new AssertionError("Could not save the user preference to SharedPreferences!");
    }
    createSession();
  }

  /** Checks if we detected at least one plane. */
  private boolean hasTrackingPlane() {
    for (Plane plane : session.getAllTrackables(Plane.class)) {
      if (plane.getTrackingState() == TrackingState.TRACKING) {
        return true;
      }
    }
    return false;
  }

  /** Update state based on the current frame's light estimation. */
  private void updateLightEstimation(LightEstimate lightEstimate, float[] viewMatrix) {
    if (lightEstimate.getState() != LightEstimate.State.VALID) {
      virtualObjectShader.setBool("u_LightEstimateIsValid", false);
      return;
    }
    virtualObjectShader.setBool("u_LightEstimateIsValid", true);

    Matrix.invertM(viewInverseMatrix, 0, viewMatrix, 0);
    virtualObjectShader.setMat4("u_ViewInverse", viewInverseMatrix);

    updateMainLight(
        lightEstimate.getEnvironmentalHdrMainLightDirection(),
        lightEstimate.getEnvironmentalHdrMainLightIntensity(),
        viewMatrix);
    updateSphericalHarmonicsCoefficients(
        lightEstimate.getEnvironmentalHdrAmbientSphericalHarmonics());
    cubemapFilter.update(lightEstimate.acquireEnvironmentalHdrCubeMap());
  }

  private void updateMainLight(float[] direction, float[] intensity, float[] viewMatrix) {
    // We need the direction in a vec4 with 0.0 as the final component to transform it to view space
    worldLightDirection[0] = direction[0];
    worldLightDirection[1] = direction[1];
    worldLightDirection[2] = direction[2];
    Matrix.multiplyMV(viewLightDirection, 0, viewMatrix, 0, worldLightDirection, 0);
    virtualObjectShader.setVec4("u_ViewLightDirection", viewLightDirection);
    virtualObjectShader.setVec3("u_LightIntensity", intensity);
  }

  private void updateSphericalHarmonicsCoefficients(float[] coefficients) {
    // Pre-multiply the spherical harmonics coefficients before passing them to the shader. The
    // constants in sphericalHarmonicFactors were derived from three terms:
    //
    // 1. The normalized spherical harmonics basis functions (y_lm)
    //
    // 2. The lambertian diffuse BRDF factor (1/pi)
    //
    // 3. A <cos> convolution. This is done to so that the resulting function outputs the irradiance
    // of all incoming light over a hemisphere for a given surface normal, which is what the shader
    // (environmental_hdr.frag) expects.
    //
    // You can read more details about the math here:
    // https://google.github.io/filament/Filament.html#annex/sphericalharmonics

    if (coefficients.length != 9 * 3) {
      throw new IllegalArgumentException(
          "The given coefficients array must be of length 27 (3 components per 9 coefficients");
    }

    // Apply each factor to every component of each coefficient
    for (int i = 0; i < 9 * 3; ++i) {
      sphericalHarmonicsCoefficients[i] = coefficients[i] * sphericalHarmonicFactors[i / 3];
    }
    virtualObjectShader.setVec3Array(
        "u_SphericalHarmonicsCoefficients", sphericalHarmonicsCoefficients);
  }

  /** Configures the session with feature settings. */
  private void configureSession() {
    Config config = session.getConfig();
    config.setLightEstimationMode(Config.LightEstimationMode.ENVIRONMENTAL_HDR);
    if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
      config.setDepthMode(Config.DepthMode.AUTOMATIC);
    } else {
      config.setDepthMode(Config.DepthMode.DISABLED);
    }
    if (instantPlacementSettings.isInstantPlacementEnabled()) {
      config.setInstantPlacementMode(InstantPlacementMode.LOCAL_Y_UP);
    } else {
      config.setInstantPlacementMode(InstantPlacementMode.DISABLED);
    }
    config.setCloudAnchorMode(Config.CloudAnchorMode.ENABLED);
    session.configure(config);
    cloudManager.setSession(session);
    hasSetTextureNames = false;
  }

  public static byte[] floatArrayToByteArray(float[] value) throws IOException {
    int intBits =  Float.floatToIntBits(value[0]);
    byte[] byteX = new byte[] {
            (byte) (intBits >> 24),
            (byte) (intBits >> 16),
            (byte) (intBits >> 8),
            (byte) (intBits) };
    intBits =  Float.floatToIntBits(value[1]);
    byte[] byteY = new byte[] {
            (byte) (intBits >> 24),
            (byte) (intBits >> 16),
            (byte) (intBits >> 8),
            (byte) (intBits) };

    ByteArrayOutputStream outputStream = new ByteArrayOutputStream( );
    outputStream.write(byteX);
    outputStream.write(byteY);
    return outputStream.toByteArray();
  }

  public static float[] byteBufferToFloatArray(ByteBuffer bytes) {
    int intBitsX = bytes.get() << 24
            | (bytes.get() & 0xFF) << 16
            | (bytes.get() & 0xFF) << 8
            | (bytes.get() & 0xFF);
    int intBitsY = bytes.get() << 24
            | (bytes.get() & 0xFF) << 16
            | (bytes.get() & 0xFF) << 8
            | (bytes.get() & 0xFF);
    return new float[]{Float.intBitsToFloat(intBitsX), Float.intBitsToFloat(intBitsY)};
  }

  public static byte[] transformPointCloud(FloatBuffer pcd, float[] viewMatrix){
    float[] PCD = new float[pcd.capacity()];
    pcd.get(PCD);
    float[] transformedPCD = new float[pcd.capacity()];
    float[] xyz = new float[4];
    float[] transformed_xyz = new float[4];
    Log.d("yunho", Integer.toString(pcd.capacity()));
    for(int i = 0; i < pcd.capacity()/4; i++){
        for(int j = 0; j < 3; j ++){
          xyz[j] = PCD[i*4 + j];
        }
        xyz[3] = 1;
        Log.d("yunho-before", Arrays.toString(xyz));
        Matrix.multiplyMV(transformed_xyz, 0, viewMatrix, 0, xyz, 0);
        Log.d("yunho-after", Arrays.toString(transformed_xyz));
        for(int j = 0; j < 3; j ++){
          transformedPCD[i*4 + j] = transformed_xyz[j];
        }
        transformedPCD[i*4 + 3] = pcd.get(i*4 + 3);
    }
//    Log.d("yunho-PCD", Arrays.toString(transformedPCD));
    ByteBuffer byteBuffer = ByteBuffer.allocate(transformedPCD.length * 4);
    byteBuffer.asFloatBuffer().put(transformedPCD);
    byte[] bytearray = byteBuffer.array();
//    Log.d("yunho-byte", Arrays.toString(bytearray));
    return bytearray;
  }

//  public static byte[] pointCloudToByteArray(FloatBuffer pcd){
//    ByteBuffer byteBuffer = ByteBuffer.allocate(pcd.capacity() * 4);
//    byteBuffer.asFloatBuffer().put(pcd);
//    byte[] bytearray = byteBuffer.array();
//    return bytearray;
//  }
}
