import * as React from "react";
import * as ReactDOM from "react-dom/client";

function Widget({ model }) {
  const [count, setCount] = React.useState(model.get("count"));

  React.useEffect(() => {
    const onChange = () => setCount(model.get("count"));
    model.on("change:count", onChange);
    return () => model.off("change:count", onChange);
  }, [model]);

  const handleClick = () => {
    model.set("count", count + 1);
    model.save_changes();
  };

  return (
    <div style={{ padding: "16px", fontFamily: "sans-serif" }}>
      <h3>quantem.widget</h3>
      <p>Count: {count}</p>
      <button onClick={handleClick}>Increment</button>
    </div>
  );
}

function render({ model, el }) {
  const root = ReactDOM.createRoot(el);
  root.render(<Widget model={model} />);
  return () => root.unmount();
}

export default { render };
