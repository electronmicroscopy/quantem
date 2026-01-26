// Placeholder for Show3D widget
// TODO: Implement 3D volume viewer widget

import * as React from "react";
import { createRender } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

function Show3D() {
  return (
    <Box sx={{ p: 2, border: "1px solid #444", borderRadius: 1 }}>
      <Typography>Show3D - Coming Soon</Typography>
    </Box>
  );
}

export const render = createRender(Show3D);
