import React, { useState } from 'react';
import { BarChart3, X } from 'lucide-react';

interface MockupButtonProps {
  onMockupClick: (mockupData: any) => void;
}

const MockupButton: React.FC<MockupButtonProps> = ({ onMockupClick }) => {
  const [showModal, setShowModal] = useState(false);

  const mockupExamples = [
    {
      id: 1,
      title: "Singapore Import/Export Analysis",
      description: "Combined bar and line chart showing monthly import/export data",
      question: "create a chart of total export and import at singapore warehouse in 2024 by each month, import use line chart and export use bar chart",
      answer: "The chart displays the monthly import and export quantities for the Singapore warehouse in 2024. Import data is sourced from df1 using 'INBOUND_DATE' and 'NET_QUANTITY_MT', while export data is sourced from df2 using 'OUTBOUND_DATE' and 'NET_QUANTITY_MT'. Both datasets were filtered for the 'SINGAPORE-WAREHOUSE' and the year 2024, then grouped by month to sum the 'NET_QUANTITY_MT'. The 'INBOUND_DATE' and 'OUTBOUND_DATE' columns were converted to datetime objects and then formatted to 'YYYY-MM' for consistent monthly grouping. The final chart combines these two datasets, showing imports as a line to highlight trends and exports as bars for clear monthly comparisons.",
      chart: "https://quickchart.io/chart?c={%22type%22:%22bar%22,%22data%22:{%22labels%22:[%222024-01%22,%222024-02%22,%222024-03%22,%222024-04%22,%222024-05%22,%222024-06%22,%222024-07%22,%222024-08%22,%222024-09%22,%222024-10%22,%222024-11%22,%222024-12%22],%22datasets%22:[{%22borderColor%22:%22%235bc0de%22,%22type%22:%22line%22,%22data%22:[11612.5,10700,11425,10325,11000,10875,12050,12500,11900,13000,12200,10500],%22fill%22:false,%22label%22:%22Total%20Import%20(MT)%22},{%22label%22:%22Total%20Export%20(MT)%22,%22data%22:[8030.35,10046.8,11107.9,8819.18,10502,12455.7,12329.3,13324,12277.8,15154.1,14412.8,11252.1],%22backgroundColor%22:%22%23d9534f%22}]}}"
    },
    {
      id: 2,
      title: "Top 10 Materials Analysis",
      description: "Pie chart showing top 10 imported materials in January 2024",
      question: "create a pie chart for top 10 import on january in 2024 by material",
      answer: "The pie chart displays the top 10 materials by net quantity (MT) for imports that occurred in January 2024. First, the 'INBOUND_DATE' column in df1 was converted to datetime objects. Then, the DataFrame was filtered to include only imports in January 2024. Finally, the data was grouped by 'MATERIAL_NAME' and the 'NET_QUANTITY_MT' was summed, with the top 10 materials selected for visualization.",
      chart: "https://quickchart.io/chart?c={%22type%22:%22pie%22,%22data%22:{%22labels%22:[%22MAT-0332%22,%22MAT-0145%22,%22MAT-0001%22,%22MAT-0319%22,%22MAT-0335%22,%22MAT-0316%22,%22MAT-0321%22,%22MAT-0389%22,%22MAT-0382%22,%22MAT-0413%22],%22datasets%22:[{%22data%22:[2967,2728.5,1708.5,1336.5,1237.5,1188,1188,892.5,891,841.5],%22backgroundColor%22:[%22%23FF6384%22,%22%2336A2EB%22,%22%23FFCE56%22,%22%234BC0C0%22,%22%239966FF%22,%22%23FF9F40%22,%22%23E7E9ED%22,%22%238AC926%22,%22%231982C4%22,%22%236A4C93%22]}]}}"
    },
    {
      id: 3,
      title: "First Half 2024 Radar Analysis",
      description: "Radar chart comparing import/export patterns for first 6 months",
      question: "create radar chart for import and export in first half of 2024 by month",
      answer: "The radar chart displays the total net quantity in metric tons for both imports and exports during the first six months of 2024. The data is aggregated monthly, providing a clear comparison of inbound and outbound material flow over time. Each spoke of the radar chart represents a month, with the length of the spoke indicating the total quantity for that month. The red line represents the total import quantity, and the blue line represents the total export quantity.",
      chart: "https://quickchart.io/chart?c={%22type%22:%22radar%22,%22data%22:{%22labels%22:[%22January%22,%22February%22,%22March%22,%22April%22,%22May%22,%22June%22],%22datasets%22:[{%22data%22:[31892.125,23598.108,49456.813,37863.85,42019.03,26488.425],%22backgroundColor%22:%22rgba(255,%2099,%20132,%200.2)%22,%22label%22:%22Total%20Import%20Quantity%20(MT)%22,%22borderWidth%22:1,%22borderColor%22:%22rgba(255,%2099,%20132,%201)%22},{%22data%22:[33718.125,24072.445,31546.415,29881.785,33264.76,33465.855],%22backgroundColor%22:%22rgba(54,%20162,%20235,%200.2)%22,%22label%22:%22Total%20Export%20Quantity%20(MT)%22,%22borderColor%22:%22rgba(54,%20162,%20235,%201)%22,%22borderWidth%22:1}]}}"
    }
  ];

  const handleMockupSelect = (mockup: any) => {
    onMockupClick(mockup);
    setShowModal(false);
  };

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="fixed top-6 left-6 px-5 py-3 exxon-gradient text-white rounded-full shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 z-10 group flex items-center gap-2"
        title="Query Suggestions"
      >
        <BarChart3 className="w-4 h-4" />
        <span className="text-sm font-semibold">Query Suggestions</span>
        <div className="absolute -bottom-12 left-0 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
          Choose from 3 suggested queries
        </div>
      </button>

      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="exxon-card rounded-2xl p-6 max-w-4xl w-full exxon-shadow relative max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowModal(false)}
              className="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="text-center mb-6">
              <div className="w-16 h-16 exxon-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-exxon-dark-blue mb-2">Suggested Query Examples</h2>
              <p className="text-exxon-text">Select from our suggested queries with real AI responses to explore analytics capabilities</p>
              <div className="bg-gradient-to-r from-exxon-red to-exxon-light-blue h-1 rounded-full w-20 mx-auto mt-3"></div>
            </div>

            <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-3">
              {mockupExamples.map((mockup) => (
                <div
                  key={mockup.id}
                  className="exxon-card border border-gray-200 rounded-xl p-5 hover:shadow-lg transition-all duration-200 cursor-pointer group hover:border-exxon-light-blue"
                  onClick={() => handleMockupSelect(mockup)}
                >
                  <div className="flex items-start gap-3 mb-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-exxon-red to-exxon-light-blue rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-white font-bold text-sm">{mockup.id}</span>
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-exxon-dark-blue text-sm mb-1 group-hover:text-exxon-light-blue transition-colors">
                        {mockup.title}
                      </h3>
                      <p className="text-exxon-text text-xs leading-relaxed">
                        {mockup.description}
                      </p>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-exxon-red/10 to-exxon-light-blue/10 border border-exxon-light-blue/20 rounded-lg p-3 mt-3">
                    <p className="text-exxon-dark-blue text-xs font-medium truncate">
                      "{mockup.question.substring(0, 60)}..."
                    </p>
                  </div>

                  <div className="mt-3 flex items-center justify-between">
                    <span className="text-exxon-text/60 text-xs">Click to try suggestion</span>
                    <div className="w-4 h-4 border-2 border-exxon-light-blue rounded-full flex items-center justify-center group-hover:bg-exxon-light-blue transition-colors">
                      <div className="w-1.5 h-1.5 bg-exxon-light-blue rounded-full group-hover:bg-white transition-colors"></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 text-center">
              <div className="bg-gradient-to-r from-exxon-red/10 to-exxon-light-blue/10 border border-exxon-light-blue/20 rounded-xl p-4">
                <p className="text-exxon-dark-blue text-sm font-semibold flex items-center justify-center gap-2 mb-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  Suggested Queries with Real AI Responses
                </p>
                <p className="text-exxon-text/70 text-sm">
                  These are suggested query examples that demonstrate our AI's ability to analyze warehouse data and generate real insights with visualizations
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default MockupButton;