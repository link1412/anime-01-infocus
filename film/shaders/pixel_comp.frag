#version 410
// Nearest-neighbour blocks on a fixed grid: no filtering, no blending, display-referred output.
out vec4 o;
uniform sampler2D uTex;
uniform float uBlock, uY0, uHeight;
void main(){
  vec2 w=vec2(gl_FragCoord.x,uHeight-gl_FragCoord.y);
  ivec2 p=ivec2(floor(w.x/uBlock),floor((w.y-uY0)/uBlock));
  o=vec4(texelFetch(uTex,p,0).rgb,1.);
}
