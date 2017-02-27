#include <python2.7/Python.h>

#include "../src/network.h"
#include "../src/region_layer.h"
#include "../src/cost_layer.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/box.h"
#include "../src/demo.h"
#include "../src/option_list.h"

int get_detections(int *class_list, float *prob_list, int num, float thresh, float **probs, int classes)
{
    int i;
	int return_num = 0;

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            //printf("%s: %.0f%%\n", names[class], prob*100);
            printf("%d: %.3f\n", class, prob);

			class_list[return_num] = class;
			prob_list[return_num] = prob;
			return_num += 1; 
        }
    }
	return return_num;
}

static void delet_net(void* ptr)
{
    network *p_Net = (network *)ptr;	
	if (p_Net)
	{
		free_network(*p_Net);
		p_Net = NULL;
	}
	return;
}

PyObject* init_model(PyObject *self, PyObject *args)
{
	char *datacfg = NULL;
	char *cfgfile = NULL;
	char *weightfile = NULL;
	int gpu_index = 0;

    if (!PyArg_ParseTuple(args, "sssi", &datacfg, &cfgfile, &weightfile, &gpu_index))
	{
		printf("init model parse args error\n");
		Py_RETURN_NONE;
	}

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

#ifdef GPU
	printf("use gpu %d\n", gpu_index);
    cuda_set_device(gpu_index);
#endif
    //network net = parse_network_cfg(cfgfile);
	network *net = calloc(1, sizeof(network)); 
    *net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(net, weightfile);
    }
    set_batch_network(net, 1);

	return PyCObject_FromVoidPtr(net, delet_net);
}

PyObject* my_predictor(PyObject *self, PyObject *args)
{
	PyObject *pyNet = NULL;
	char *filename = NULL;
	float thresh = 0.0;
	float hier_thresh = 0.0;

    if (!PyArg_ParseTuple(args, "Osff"
							, &pyNet
							, &filename
							, &thresh
							, &hier_thresh
							))
	{
		printf("predect parse args error\n");
		Py_RETURN_NONE;
	}

	void *temp = PyCObject_AsVoidPtr(pyNet);
	if (!temp)
		Py_RETURN_NONE;

    network *pNet = (network *)temp;
	if (!pNet)
		Py_RETURN_NONE;

	network net  = *pNet;
    srand(2222222);
    clock_t time;
    int j;
    float nms=.4;
    image im = load_image_color(filename,0,0);
    image sized = resize_image(im, net.w, net.h);
    layer l = net.layers[net.n-1];
    
    // save predict results
    int* class_list = calloc(l.w*l.h*l.n, sizeof(int));
    float* prob_list = calloc(l.w*l.h*l.n, sizeof(float));
	int return_num = 0;
	PyObject* resList = PyList_New(0);
    
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
    
    float *X = sized.data;
    time=clock();
    network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);

    if (l.softmax_tree && nms)
	{
			printf("soft max \n");
			do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

	}
    else if (nms)
	{
			printf("not soft max \n");
			do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	}
    return_num = get_detections(class_list, prob_list, l.w*l.h*l.n, thresh, probs, l.classes);
    //draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
    //save_image(im, "predictions");
    //show_image(im, "predictions");
    
    int i;
    for( i = 0; i < return_num; i++)
    {
    	if (class_list[i] != 0)
		{
    		//printf("class list %d:%d\n", i, class_list[i], prob_list[i]);
			PyObject* obj = Py_BuildValue("if", class_list[i], prob_list[i]);
			PyList_Append(resList, obj);
			Py_DECREF(obj);
		}
    }
    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.w*l.h*l.n);
    
    // free
    free(class_list);
    free(prob_list);
    
	return resList;
}

PyObject* MyTest(PyObject *self, PyObject* args)
{
	printf("just test \n");
	Py_RETURN_NONE;
}

static PyMethodDef _CInterface[] = {
		{"test", MyTest, METH_VARARGS, "just test"},
		{"init_model", init_model, METH_VARARGS, "init model"},
		{"predict", my_predictor, METH_VARARGS, "predict one img"},
		{NULL, NULL, 0, NULL}
};

void initDarknetPre()
{
	PyObject* obj = NULL;
	//obj =  Py_InitModule("CIndex", _CIndex);
	obj =  Py_InitModule("DarknetPre", _CInterface);
}
