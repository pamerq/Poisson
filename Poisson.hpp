 #include "Essentials/Essentials.hpp"
#include "Essentials/Distances.hpp"

#include <assert.h>
#include <math.h>
#include <random>

namespace poisson{

    const brahand::NavigationMap neighbors2DMap {
    	{-2,-2,0},{-2,-1,0},{-2,0,0},{-2,1,0},{-1,2,0},
    	{-1,-2,0},{-1,-1,0},{-1,0,0},{-1,1,0},{-1,2,0},
    	{0,-2,0},{0,-1,0},{0,0,0},{0,1,0},{0,2,0},
    	{1,-2,0},{1,-1,0},{1,0,0},{1,1,0},{1,2,0},
    	{2,-2,0},{2,-1,0},{2,0,0},{2,1,0},{1,2,0}
    };

    const brahand::NavigationMap neighbors3DMap {
    	{-2,-2,-1},{-2,-1,-1},{-2,0,-1},{-2,1,-1},{-1,2,-1},
    	{-1,-2,-1},{-1,-1,-1},{-1,0,-1},{-1,1,-1},{-1,2,-1},
    	{0,-2,-1},{0,-1,-1},{0,0,-1},{0,1,-1},{0,2,-1},
    	{1,-2,-1},{1,-1,-1},{1,0,-1},{1,1,-1},{1,2,-1},
    	{2,-2,-1},{2,-1,-1},{2,0,-1},{2,1,-1},{1,2,-1},

    	{-2,-2,0},{-2,-1,0},{-2,0,0},{-2,1,0},{-1,2,0},
    	{-1,-2,0},{-1,-1,0},{-1,0,0},{-1,1,0},{-1,2,0},
    	{0,-2,0},{0,-1,0},{0,0,0},{0,1,0},{0,2,0},
    	{1,-2,0},{1,-1,0},{1,0,0},{1,1,0},{1,2,0},
    	{2,-2,0},{2,-1,0},{2,0,0},{2,1,0},{1,2,0},

    	{-2,-2,1},{-2,-1,1},{-2,0,1},{-2,1,1},{-1,2,1},
    	{-1,-2,1},{-1,-1,1},{-1,0,1},{-1,1,1},{-1,2,1},
    	{0,-2,1},{0,-1,1},{0,0,1},{0,1,1},{0,2,1},
    	{1,-2,1},{1,-1,1},{1,0,1},{1,1,1},{1,2,1},
    	{2,-2,1},{2,-1,1},{2,0,1},{2,1,1},{1,2,1}
    };

    struct Grid {
    	brahand::usize rows, cols, layers;
    	std::vector< std::vector< brahand::ulsize> > cells;

    	Grid() = default;

    	std::vector< brahand::ulsize>& operator()(brahand::usize i, brahand::usize j, brahand::usize k){
            assert(i >= 0 && i < rows && j >= 0 && j < cols && k >= 0 && k < layers);
    		brahand::ulsize index = (k * cols  * rows) + (i * cols) + j;
    		return cells[index];
    	}

    	void reserve(brahand::usize rows, brahand::usize cols, brahand::usize layers){
    		this->rows	= rows;
    		this->cols	= cols;
    		this->layers = layers;
    		cells		= std::vector< std::vector< brahand::ulsize> >( this->rows * this->cols * this->layers );
    	}

        void print(){
            printf("Grid (%d,%d,%d)\n", rows, cols, layers);
            for(int i = 0 ; i < cells.size() ; ++i){
                printf("[%d: %lu]: ", i, cells[i].size() );
                for( int j = 0 ; j < cells[i].size() ; ++j){
                    std::cout << cells[i][j] << " ";
                }
                printf("\n");
            }

        }
    };

    struct EuclideanCoordinate{
        float x,y,z;
    };

    template <class T>
    T chooseRandom(T lower, T upper, std::mt19937 randomEngine ){
    	std::uniform_int_distribution<T> processingListDistribution(lower, upper);
    	return processingListDistribution(randomEngine);
    }

    class PoissonSampler {
    private:
        brahand::ImageSize size;
        brahand::IndicesArray coordinates;
        float cellSize;
        brahand::ulsize k;
        brahand::FastMarchingPropagation density;

        Grid poissonGrid, imageGrid;
        std::vector<brahand::ulsize> validCellsIds;
        std::vector<brahand::ulsize> processingList, outputList;

        std::random_device randomDevice;
        std::mt19937 randomEngine{randomDevice()};

        EuclideanCoordinate imageCoordinateToCellCoordinate(brahand::ulsize coordinate) {
            brahand::ulsize idx, row, col, layer;
            idx = coordinate;
            layer = idx / (size.width * size.height); // z
            idx -= (layer * size.width * size.height);
            row = idx / size.width; // y
            col = idx % size.width; // x

            assert(col >= 0 && col < this->size.width && row >= 0 && row < this->size.height && layer >= 0 && layer < this->size.depth);

    		float i = (float) (floorf((float)row / this->cellSize));
    		float j = (float) (floorf((float)col / this->cellSize));
    		float k = (float) (floorf((float)layer / this->cellSize));

    		return EuclideanCoordinate{i,j,k};
    	}

    public:
        PoissonSampler() = delete;
        PoissonSampler(brahand::ImageSize size, float cellSize, brahand::ulsize k, brahand::IndicesArray coordinates, brahand::FastMarchingPropagation density, brahand::IndicesArray existingSamples){

            this->k = k;
            this->density = density;
            this->cellSize = cellSize;
            this->size = size;
            this->coordinates = coordinates;

            brahand::usize gridCols = ceilf((float)this->size.width / cellSize);
            brahand::usize gridRows = ceilf((float)this->size.height / cellSize);
            brahand::usize gridLayers = ceilf((float)this->size.depth / cellSize);

            this->poissonGrid.reserve(gridRows, gridCols, gridLayers);
    		this->imageGrid.reserve(gridRows, gridCols, gridLayers);

            /// Read EuclideanCoordinates and assign it to imageGrid;
    		for(brahand::ulsize i = 0 ; i < coordinates.count ; ++i){
                auto coordinateIndex = this->coordinates[i];
    			auto gridCoordinate = imageCoordinateToCellCoordinate(coordinateIndex);
    			this->imageGrid(gridCoordinate.x, gridCoordinate.y, gridCoordinate.z).push_back(coordinateIndex);
    		}
            // imageGrid.print();

            /// Get index of valid cells (non-empty) from imageGrid
    		for (brahand::ulsize imgIndex = 0 ; imgIndex < this->imageGrid.cells.size() ; ++imgIndex) {
    			if( this->imageGrid.cells[imgIndex].size() > 0 ){ this->validCellsIds.push_back(imgIndex); }
    		}

            for(brahand::ulsize i = 0 ; i < existingSamples.count ; ++i ){
                this->outputList.push_back(existingSamples[i]);
                EuclideanCoordinate gridCoordinate = imageCoordinateToCellCoordinate(existingSamples[i]);
                this->poissonGrid(gridCoordinate.x, gridCoordinate.y, gridCoordinate.z).push_back(existingSamples[i]);
            }

            brahand::ulsize initialPoint;

            /// Choose a random point from image's valid cells that is not in conflict with already inserted points from edge density
            do{
                brahand::ulsize randomIndex = chooseRandom<brahand::ulsize>(0, this->validCellsIds.size()-1, this->randomEngine);
                initialPoint = imageGrid.cells[this->validCellsIds[randomIndex]][0]; // first one
            } while( coordinateIsInConflictWithNeighbors(initialPoint, iterateNeighbors(imageCoordinateToCellCoordinate(initialPoint), this->poissonGrid) ) );


    		/// Locate the choosen point into the grid coordinates.
    		EuclideanCoordinate gridCoodinate = imageCoordinateToCellCoordinate(initialPoint);

            /// Add it to grid cells, processing list and output list

    		this->processingList.push_back(initialPoint);
    		this->poissonGrid(gridCoodinate.x, gridCoodinate.y, gridCoodinate.z).push_back(initialPoint);
            this->outputList.push_back(initialPoint);

            // poissonGrid.print();
        }

        PoissonSampler(brahand::ImageSize size, float cellSize, brahand::ulsize k, brahand::IndicesArray coordinates, brahand::FastMarchingPropagation density){
            this->k = k;
            this->density = density;
            this->cellSize = cellSize;
            this->size = size;
            this->coordinates = coordinates;

            brahand::usize gridCols = ceilf((float)this->size.width / cellSize);
            brahand::usize gridRows = ceilf((float)this->size.height / cellSize);
            brahand::usize gridLayers = ceilf((float)this->size.depth / cellSize);

            this->poissonGrid.reserve(gridRows, gridCols, gridLayers);
    		this->imageGrid.reserve(gridRows, gridCols, gridLayers);

            /// Read EuclideanCoordinates and assign it to imageGrid;
    		for(brahand::ulsize i = 0 ; i < coordinates.count ; ++i){
                auto coordinateIndex = this->coordinates[i];
    			auto gridCoordinate = imageCoordinateToCellCoordinate(coordinateIndex);
    			this->imageGrid(gridCoordinate.x, gridCoordinate.y, gridCoordinate.z).push_back(coordinateIndex);
    		}
            // imageGrid.print();

            /// Get index of valid cells (non-empty) from imageGrid
    		for (brahand::ulsize imgIndex = 0 ; imgIndex < this->imageGrid.cells.size() ; ++imgIndex) {
    			if( this->imageGrid.cells[imgIndex].size() > 0 ){ this->validCellsIds.push_back(imgIndex); }
    		}

            // printf("Valid cells ids (%lu): ", validCellsIds.size());
            // for(int i = 0 ;i  < validCellsIds.size() ; ++i){
            //     std::cout  << validCellsIds[i] << " ";
            // }
            // printf("\n");

            /// Choose a random point from image's valid cells
    		brahand::ulsize randomIndex = chooseRandom<brahand::ulsize>(0, this->validCellsIds.size()-1, this->randomEngine);
    		brahand::ulsize initialPoint = imageGrid.cells[this->validCellsIds[randomIndex]][0]; // first one

            // printf("Initial point: %d\n", initialPoint);

    		/// Locate the choosen point into the grid coordinates.
    		EuclideanCoordinate gridCoodinate = imageCoordinateToCellCoordinate(initialPoint);

            /// Add it to grid cells, processing list and output list
    		this->outputList.push_back(initialPoint);
    		this->processingList.push_back(initialPoint);
    		this->poissonGrid(gridCoodinate.x, gridCoodinate.y, gridCoodinate.z).push_back(initialPoint);

            // poissonGrid.print();
        }

        std::vector<brahand::ulsize> iterateNeighbors(EuclideanCoordinate center, Grid grid){
    		std::vector<brahand::ulsize> neighborsArray;

    		brahand::NavigationMap map = (size.depth == 1) ? neighbors2DMap : neighbors3DMap;

    		for( auto navigation :  map ){
    			int x = (int)(center.x + (int)(navigation[0]));
    			int y = (int)(center.y + (int)(navigation[1]));
    			int z = (int)(center.z + (int)(navigation[2]));

    			if( x  >=  0 && x < grid.rows && y >= 0 && y < grid.cols && z >= 0 && z < grid.layers){
    				std::vector< brahand::ulsize> coordinates = grid(x, y, z);
    				if( coordinates.size() > 0 ){
    					neighborsArray.insert (neighborsArray.end(), coordinates.begin(), coordinates.end());
    				}
    			}
    		}
    		return neighborsArray;
    	}

        bool coordinateIsInConflictWithNeighbors(brahand::ulsize coordinate, std::vector<brahand::ulsize> neighborsArray ){

            brahand::ulsize idx, row, col, layer;
            idx = coordinate;
            layer = idx / (size.width * size.height); // z
            idx -= (layer * size.width * size.height);
            row = idx / size.width; // y
            col = idx % size.width; // x

    		if(std::any_of(neighborsArray.begin() , neighborsArray.end(), [&](brahand::ulsize n){

                brahand::ulsize nidx, nrow, ncol, nlayer;
                nidx = n;
                nlayer = nidx / (size.width * size.height); // z
                nidx -= (nlayer * size.width * size.height);
                nrow = nidx / size.width; // y
                ncol = nidx % size.width; // x

                float d = brahand::euclideanDistance<float>((float)col, (float)row, (float)layer,(float)ncol, (float)nrow, (float)nlayer);
                // printf("%d [%d] - %d [%d]\n", density.propagation[coordinate].density,coordinate,  density.propagation[n].density, n);
    			return d < std::min(density.propagation[coordinate].density, density.propagation[n].density ) ;
    		}) ){ return true; }

    		return false;
    	}


        brahand::IndicesArray sampling(){
            while( this->processingList.size() > 0 ){
                /// Choose a random point from the processing list.
    			brahand::ulsize randomIndex = chooseRandom<brahand::ulsize>(0, processingList.size()-1, this->randomEngine);

    			brahand::ulsize randomPoint = processingList[randomIndex];
    			EuclideanCoordinate randomPointGridCoordinate = imageCoordinateToCellCoordinate(randomPoint);

                // printf("Random point: %d: %f,%f,%f\n", randomPoint, randomPointGridCoordinate.x, randomPointGridCoordinate.y, randomPointGridCoordinate.z);

                /// For this point, generate up to k points, randomly selected from the annulus surrounding the point.
    			std::vector<brahand::ulsize> neighbors = iterateNeighbors(randomPointGridCoordinate, this->imageGrid);

                // printf("\t Neighbors (%lu): ", neighbors.size());
                // for(int i = 0 ; i < neighbors.size() ; ++i){
                //     std::cout << neighbors[i] << " ";
                // }
                // printf("\n");

                /// If k is not enough, change it.
                if(neighbors.size() <= this->k){ this->k = neighbors.size(); }

                /// Choose k samples from mergedArray. Unsort get k first
                std::shuffle(neighbors.begin(), neighbors.end(), this->randomEngine);
                std::vector<brahand::ulsize> kSamples;
                kSamples.insert(kSamples.end(), neighbors.begin(), neighbors.begin() + this->k);

                // printf("\t Neighbors shuffle (%lu): ", kSamples.size());
                // for(int i = 0 ; i < kSamples.size() ; ++i){
                //     std::cout << kSamples[i] << " ";
                // }
                // printf("\n");

                for(brahand::ulsize sample : kSamples){

    				EuclideanCoordinate sampleGridCoordinate = imageCoordinateToCellCoordinate(sample);
    				std::vector<brahand::ulsize> sampleNeighborsArray = iterateNeighbors(sampleGridCoordinate, this->poissonGrid);

    				if( !coordinateIsInConflictWithNeighbors(sample, sampleNeighborsArray) ){
                        // printf(">> %d\n", sample);
    					this->outputList.push_back(sample);
    					this->processingList.push_back(sample);
    					this->poissonGrid(sampleGridCoordinate.x, sampleGridCoordinate.y, sampleGridCoordinate.z ).push_back(sample);
    				}
    			}
    			processingList.erase(processingList.begin() + randomIndex);
            }

            brahand::IndicesArray samples =  brahand::IndicesArray(outputList.size());
    		for(brahand::ulsize i = 0 ; i < outputList.size() ; ++i){ samples[i] = outputList[i]; }
    		return samples;
        }
    };
	
	 	 
	brahand::IndicesArray addMarginPoints(brahand::ImageSize size, float porcentajeSeparationWidth,float porcentajeSeparationHeight){
		
		brahand::ulsize separationWidth = (size.width*porcentajeSeparationWidth)-2;
		brahand::ulsize separationHeight = (size.height*porcentajeSeparationHeight)-2;
		if(separationWidth==-1){separationWidth=0;} 
		if(separationHeight==-1){separationHeight=0;}  
		brahand::ulsize num = (size.width * size.height) - (size.width); 
		brahand::ulsize numPointWidth =  floor((size.width-1)/(separationWidth+1))+1;
		brahand::ulsize numPointHeight =  floor((size.height-1)/(separationHeight+1))+1;
		brahand::ulsize sizePointMargin =  (numPointWidth*2)+((numPointHeight-2)*2); 
 		brahand::IndicesArray array = brahand::IndicesArray(sizePointMargin);
 		brahand::ulsize position = 0;
			 
		for( brahand::ulsize i = 0 ; i < size.width - separationWidth-1; i=i+separationWidth+1){ 
			array[position]=i; position++;
			array[position]=i+num; position++; 
		}
		array[position]=size.width - 1; position++;
		array[position]=size.width + num - 1; position++; 
 			 
      	for( brahand::ulsize i = (separationHeight+1) ; i < size.height - separationHeight - 1; i=i+separationHeight+1){ 
			array[position]=i*(size.width); position++;
			array[position]=(i*(size.width))+size.width-1; position++; 
		}  
		return array;
	}
}
