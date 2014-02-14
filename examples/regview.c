/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "libfreenect.h"

#include <pthread.h>

#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <signal.h>

pthread_t freenect_thread;
volatile sig_atomic_t die = 0;

int g_argc;
char **g_argv;

int window;

pthread_mutex_t gl_backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;

// back: owned by libfreenect (implicit for depth)
// mid: owned by callbacks, "latest frame ready"
// front: owned by GL, "currently being drawn"
uint8_t *depth_mid, *depth_front;
uint8_t *rgb_back, *rgb_mid, *rgb_front;

GLuint gl_depth_tex;
GLuint gl_rgb_tex;

freenect_context *f_ctx;
freenect_device *f_dev;
int freenect_angle = 0;
int freenect_led;

pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;
int got_rgb = 0;
int got_depth = 0;

int frame = 0;
int ftime = 0;
double fps = 0;

char *out_dir=0;
uint32_t last_timestamp = 0;
FILE *index_fp = NULL;

#define FREENECT_FRAME_W 640
#define FREENECT_FRAME_H 480

double get_time()
{
	struct timeval cur;
	gettimeofday(&cur, NULL);
	return cur.tv_sec + cur.tv_usec / 1000000.;
}

void dump_depth(FILE *fp, void *data, int data_size)
{
	fprintf(fp, "P5 %d %d 65535\n", FREENECT_FRAME_W, FREENECT_FRAME_H);
	fwrite(data, data_size, 1, fp);
}

void dump_rgb(FILE *fp, void *data, int data_size)
{
	fprintf(fp, "P6 %d %d 255\n", FREENECT_FRAME_W, FREENECT_FRAME_H);
	fwrite(data, data_size, 1, fp);
}

FILE *open_dump(char type, double cur_time, uint32_t timestamp, int data_size, const char *extension)
{
	char *fn = malloc(strlen(out_dir) + 50);
	sprintf(fn, "%c-%f-%u.%s", type, cur_time, timestamp, extension);
	fprintf(index_fp, "%s\n", fn);
	sprintf(fn, "%s/%c-%f-%u.%s", out_dir, type, cur_time, timestamp, extension);
	FILE* fp = fopen(fn, "wb");
	if (!fp) {
		printf("Error: Cannot open file [%s]\n", fn);
		exit(1);
	}
	//printf("%s\n", fn);
	free(fn);
	return fp;
}

FILE *open_index(const char *fn)
{
    FILE *fp = fopen(fn, "r");
    if (fp) {
        fclose(fp);
        printf("Error: Index already exists, to avoid overwriting "
               "use a different directory.\n");
        return 0;
    }
    fp = fopen(fn, "wb");
    if (!fp) {
        printf("Error: Cannot open file [%s]\n", fn);
        return 0;
    }
    return fp;
}

void dump(char type, uint32_t timestamp, void *data, int data_size)
{
	// timestamp can be at most 10 characters, we have a few extra
	double cur_time = get_time();
	FILE *fp;
	last_timestamp = timestamp;
	switch (type) {
		case 'd':
			fp = open_dump(type, cur_time, timestamp, data_size, "pgm");
			dump_depth(fp, data, data_size);
			fclose(fp);
			break;
		case 'r':
			fp = open_dump(type, cur_time, timestamp, data_size, "ppm");
			dump_rgb(fp, data, data_size);
			fclose(fp);
			break;
		case 'a':
			fp = open_dump(type, cur_time, timestamp, data_size, "dump");
			fwrite(data, data_size, 1, fp);
			fclose(fp);
			break;
	}
}

void idle()
{
	pthread_mutex_lock(&gl_backbuf_mutex);

	// When using YUV_RGB mode, RGB frames only arrive at 15Hz, so we shouldn't force them to draw in lock-step.
	// However, this is CPU/GPU intensive when we are receiving frames in lockstep.
	while (!got_depth && !got_rgb) {
		pthread_cond_wait(&gl_frame_cond, &gl_backbuf_mutex);
	}

	if (!got_depth || !got_rgb) {
		pthread_mutex_unlock(&gl_backbuf_mutex);
		return;
	}
	pthread_mutex_unlock(&gl_backbuf_mutex);
	glutPostRedisplay();
}

void DrawGLScene() {
	uint8_t *tmp;

	pthread_mutex_lock(&gl_backbuf_mutex);
	if (got_depth) {
		tmp = depth_front;
		depth_front = depth_mid;
		depth_mid = tmp;
		got_depth = 0;
	}
	if (got_rgb) {
		tmp = rgb_front;
		rgb_front = rgb_mid;
		rgb_mid = tmp;
		got_rgb = 0;
	}
	pthread_mutex_unlock(&gl_backbuf_mutex);

	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_front);

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glTexCoord2f(0, 0); glVertex3f(0,0,0);
	glTexCoord2f(1, 0); glVertex3f(640,0,0);
	glTexCoord2f(1, 1); glVertex3f(640,480,0);
	glTexCoord2f(0, 1); glVertex3f(0,480,0);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, depth_front);

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glTexCoord2f(0, 0); glVertex3f(0,0,0);
	glTexCoord2f(1, 0); glVertex3f(640,0,0);
	glTexCoord2f(1, 1); glVertex3f(640,480,0);
	glTexCoord2f(0, 1); glVertex3f(0,480,0);
	glEnd();

	glutSwapBuffers();

	frame++;
	if (frame % 30 == 0) {
		int ms = glutGet(GLUT_ELAPSED_TIME);
		fps = 30.0/((ms-ftime)/1000.0);
		ftime = ms;
	}
}

void keyPressed(unsigned char key, int x, int y)
{
	if (key == 27 || key == 'q') {
		die = 1;
		pthread_join(freenect_thread, NULL);
		pthread_cond_signal(&gl_frame_cond);
		glutDestroyWindow(window);
		free(depth_mid);
		free(depth_front);
		free(rgb_back);
		free(rgb_mid);
		free(rgb_front);
		// Not pthread_exit because OSX leaves a thread lying around and doesn't exit
		exit(0);
	}
}

void ReSizeGLScene(int Width, int Height)
{
	glViewport(0,0,Width,Height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho (0, 640, 480, 0, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void InitGL(int Width, int Height)
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0);
	glDepthFunc(GL_LESS);
    glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_TEXTURE_2D);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glShadeModel(GL_FLAT);

	glGenTextures(1, &gl_depth_tex);
	glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenTextures(1, &gl_rgb_tex);
	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	ReSizeGLScene(Width, Height);
}

void *gl_threadfunc(void *arg)
{
	printf("GL thread\n");
	glutInit(&g_argc, g_argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(0, 0);

	window = glutCreateWindow("libfreenect Registration viewer");

	glutDisplayFunc(&DrawGLScene);
	glutIdleFunc(&idle);
	glutReshapeFunc(&ReSizeGLScene);
	glutKeyboardFunc(&keyPressed);

	InitGL(640, 480);

	glutMainLoop();

	return NULL;
}

uint16_t t_gamma[10000];

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
	int i;
	uint16_t *depth = (uint16_t*)v_depth;

	if (out_dir)
	  dump('d', timestamp, depth, freenect_get_current_depth_mode(dev).bytes);


	pthread_mutex_lock(&gl_backbuf_mutex);
	for (i=0; i<640*480; i++) {
		//if (depth[i] >= 2048) continue;
		int pval = t_gamma[depth[i]] / 4;
		int lb = pval & 0xff;
		depth_mid[4*i+3] = 128; // default alpha value
		if (depth[i] ==  0) depth_mid[4*i+3] = 0; // remove anything without depth value
		switch (pval>>8) {
			case 0:
				depth_mid[4*i+0] = 255;
				depth_mid[4*i+1] = 255-lb;
				depth_mid[4*i+2] = 255-lb;
				break;
			case 1:
				depth_mid[4*i+0] = 255;
				depth_mid[4*i+1] = lb;
				depth_mid[4*i+2] = 0;
				break;
			case 2:
				depth_mid[4*i+0] = 255-lb;
				depth_mid[4*i+1] = 255;
				depth_mid[4*i+2] = 0;
				break;
			case 3:
				depth_mid[4*i+0] = 0;
				depth_mid[4*i+1] = 255;
				depth_mid[4*i+2] = lb;
				break;
			case 4:
				depth_mid[4*i+0] = 0;
				depth_mid[4*i+1] = 255-lb;
				depth_mid[4*i+2] = 255;
				break;
			case 5:
				depth_mid[4*i+0] = 0;
				depth_mid[4*i+1] = 0;
				depth_mid[4*i+2] = 255-lb;
				break;
			default:
				depth_mid[4*i+0] = 0;
				depth_mid[4*i+1] = 0;
				depth_mid[4*i+2] = 0;
				depth_mid[4*i+3] = 0;
				break;
		}
	}
	got_depth++;
	pthread_cond_signal(&gl_frame_cond);
	pthread_mutex_unlock(&gl_backbuf_mutex);
}

void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
	pthread_mutex_lock(&gl_backbuf_mutex);

	// swap buffers
	assert (rgb_back == rgb);
	rgb_back = rgb_mid;
	freenect_set_video_buffer(dev, rgb_back);
	rgb_mid = (uint8_t*)rgb;

	if (out_dir)
	  dump('r', timestamp, rgb_mid, freenect_get_current_video_mode(dev).bytes);


	got_rgb++;
	pthread_cond_signal(&gl_frame_cond);
	pthread_mutex_unlock(&gl_backbuf_mutex);
}

void signal_cleanup(int num)
{
    printf("Caught signal, cleaning up\n");
    keyPressed('q',0,0);
    signal(SIGINT, signal_cleanup);
}

void *freenect_threadfunc(void *arg)
{
	freenect_set_depth_callback(f_dev, depth_cb);
	freenect_set_video_callback(f_dev, rgb_cb);
	freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
	freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));
	freenect_set_video_buffer(f_dev, rgb_back);

	freenect_start_depth(f_dev);
	freenect_start_video(f_dev);

	printf("'w'-tilt up, 's'-level, 'x'-tilt down, '0'-'6'-select LED mode, 'f'-video format\n");

	while (!die) {
		int res = freenect_process_events(f_ctx);
		if (res < 0 && res != -10) {
			printf("\nError %d received from libusb - aborting.\n",res);
			break;
		}
	}
	printf("\nshutting down streams...\n");

	freenect_stop_depth(f_dev);
	freenect_stop_video(f_dev);

	freenect_close_device(f_dev);
	freenect_shutdown(f_ctx);

	printf("-- done!\n");
	return NULL;
}

void usage()
{
	printf("Records the Kinect sensor data to a directory\nResult can be used as input to Fakenect\nUsage:\n");
	printf("  record [-h] [-ffmpeg] [-ffmpeg-opts <options>] "
		   "<target basename>\n");
	exit(0);
}

int main(int argc, char **argv)
{
	int res;

	depth_mid = (uint8_t*)malloc(640*480*4);
	depth_front = (uint8_t*)malloc(640*480*4);
	rgb_back = (uint8_t*)malloc(640*480*3);
	rgb_mid = (uint8_t*)malloc(640*480*3);
	rgb_front = (uint8_t*)malloc(640*480*3);

	printf("Kinect camera test\n");

	int i;
	for (i=0; i<10000; i++) {
		float v = i/2048.0;
		v = powf(v, 3)* 6;
		t_gamma[i] = v*6*256;
	}

	int c=1;
	while (c < argc) {
	  if (strcmp(argv[c],"-h")==0)
			usage();
		else
			out_dir = argv[c];
		c++;
	}

	//if (!out_dir)
	//	usage();

	g_argc = argc;
	g_argv = argv;

	if (freenect_init(&f_ctx, NULL) < 0) {
		printf("freenect_init() failed\n");
		return 1;
	}

	if (out_dir) {
	  mkdir(out_dir, S_IRWXU | S_IRWXG | S_IRWXO);
	  char *fn = malloc(strlen(out_dir) + 50);
	  sprintf(fn, "%s/INDEX.txt", out_dir);
	  index_fp = open_index(fn);
	  free(fn);
	  if (!index_fp) {
	    fclose(index_fp);
	    return 1;
	  }
	}

	//signal(SIGINT, signal_cleanup);

	freenect_set_log_level(f_ctx, FREENECT_LOG_ERROR);
	freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_CAMERA));

	int nr_devices = freenect_num_devices (f_ctx);
	printf ("Number of devices found: %d\n", nr_devices);

	int user_device_number = 0;

	if (nr_devices < 1) {
		freenect_shutdown(f_ctx);
		return 1;
	}

	if (freenect_open_device(f_ctx, &f_dev, user_device_number) < 0) {
		printf("Could not open device\n");
		freenect_shutdown(f_ctx);
		return 1;
	}

        if (out_dir) {
            printf("Recording to disk, skip OpenGL entirely\n");
            freenect_threadfunc(NULL);
            return 0;
        }
	res = pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
	if (res) {
		printf("pthread_create failed\n");
		freenect_shutdown(f_ctx);
		return 1;
	}

	// OS X requires GLUT to run on the main thread
	gl_threadfunc(NULL);

	return 0;
}
