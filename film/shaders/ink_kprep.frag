#version 410
// Bake-time: write the fibre conductivity of this layer into the state's alpha.
in vec2 uv; out vec4 o;
uniform sampler2D uS, uK;
uniform vec2 uCanvas, uOrigin;
uniform float uKTile;
void main(){
  vec4 c=texture(uS,uv);
  vec2 p=uOrigin+uv*uCanvas;
  float k=texture(uK,p/uKTile).r;
  o=vec4(c.rgb,k);
}
