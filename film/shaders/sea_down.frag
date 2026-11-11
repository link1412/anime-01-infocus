#version 410
// 13-tap downsample of the glare chain; the first level limits fireflies a little.
in vec2 uv; out vec4 o;
uniform sampler2D uTex;
uniform vec2 uPx;
uniform float uFirst;
vec3 s(vec2 d){
  vec3 c=texture(uTex,uv+d*uPx).rgb;
  return uFirst>.5?min(c,vec3(30000.)):c;
}
void main(){
  vec3 a=s(vec2(-2,2)),b=s(vec2(0,2)),c=s(vec2(2,2));
  vec3 d=s(vec2(-2,0)),e=s(vec2(0,0)),f=s(vec2(2,0));
  vec3 g=s(vec2(-2,-2)),h=s(vec2(0,-2)),i=s(vec2(2,-2));
  vec3 j=s(vec2(-1,1)),k=s(vec2(1,1)),l=s(vec2(-1,-1)),m=s(vec2(1,-1));
  vec3 r=e*.125+(a+c+g+i)*.03125+(b+d+f+h)*.0625+(j+k+l+m)*.125;
  o=vec4(r,1.);
}
