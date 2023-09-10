import { Route, Routes } from "react-router-dom";
import { routeConfig } from "../../config/routeConfig";

function App() {
  return (
    <div className="App">
      <Routes>
        {routeConfig.map(({ path, element }) => (
          <Route
            key={path}
            path={path}
            element={element}
          />
        ))}
      </Routes>
    </div>
  );
}

export default App;
